import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.io.*;
public class ClosureExport {
 static Model m;static PrintWriter w;static String data,cid;static int sol,order;
 static double[] Y={0,.0008,.0013,.00132,.00182,.00262};
 static String sig="(sigmal_YSZ*exp(-10300[K]/T))";
 static String source(int d){return d==2?"(cd.iloc_pcar1*Av_ca)":d==4?"(cd.iloc_panr1*Av_an)":"0[A/m^3]";}
 static String ohm(int d){return d==1||d==5?"0[W/m^3]":sig+"*(d(phil,x)^2+d(phil,y)^2)"+(d==3?"":"+"+(d==2?"sigmas_ca":"sigmas_an")+"*(d(phis,x)^2+d(phis,y)^2)");}
 static String act(int d){return d==2?source(d)+"*cd.eta_pcar1":d==4?source(d)+"*cd.eta_panr1":"0[W/m^3]";}
 static String rev(int d){return d==2?source(d)+"*T*dEeqdT_ca":"0[W/m^3]";}
 static String trace(String e,int d){return "if(up(dom)=="+d+",up("+e+"),down("+e+"))";}
 static double integ(String type,int entity,String e){
  String tag=type.equals("IntLine")?"clLine":"clArea";
  m.result().numerical(tag).set("data",data);m.result().numerical(tag).set("solnum",new int[]{sol});
  m.result().numerical(tag).selection().set(new int[]{entity});m.result().numerical(tag).set("expr",new String[]{e});
  m.result().numerical(tag).set("intorder",order);return m.result().numerical(tag).getReal()[0][0];
 }
 static void record(String kind,int dom,int boundary,String metric,String unit,String e){
  try{double v=integ(kind.equals("volume")?"IntSurface":"IntLine",kind.equals("volume")?dom:boundary,e);
   w.printf(Locale.US,"%s,%d,%d,%s,%d,%d,%s,%s,%.17g,ok%n",cid,sol,order,kind,dom,boundary,metric,unit,v);}
  catch(Throwable err){w.printf("%s,%d,%d,%s,%d,%d,%s,%s,NaN,error%n",cid,sol,order,kind,dom,boundary,metric,unit);System.out.println("FAILED|"+metric+"|"+dom+"|"+boundary+"|"+e+"|"+err.getMessage());}
  w.flush();
 }
 static void dumpPoints(String path)throws Exception{
  PrintWriter p=new PrintWriter(path,"UTF-8");p.println("case,solnum,domain,x,y,metric,value");
  for(int d=1;d<=5;d++){
   double[][] xy=new double[2][40];int z=0;for(int ix=0;ix<8;ix++)for(int iy=0;iy<5;iy++){xy[0][z]=.1*(ix+.5)/8;xy[1][z]=Y[d-1]+(Y[d]-Y[d-1])*(iy+.5)/5;z++;}
   m.result().numerical("clProbe").set("data",data);m.result().numerical("clProbe").set("solnum",new int[]{sol});m.result().numerical("clProbe").set("coord",xy);
   LinkedHashMap<String,String> es=new LinkedHashMap<>();es.put("T","T");es.put("V_cell","V_cell");es.put("t_s",data.equals("dset1")?"0[s]":"t");es.put("htQ","ht.Q");es.put("Qohm",ohm(d));es.put("Qactivation",act(d));es.put("Qreversible",rev(d));
   es.put("Qclosure","ht.Q-("+ohm(d)+")-("+act(d)+")-("+rev(d)+")");
   es.put("Tx","d(T,x)");es.put("Ty","d(T,y)");es.put("Tt","d(T,t)");es.put("rho","ht.rho");
   if(d==2||d==4){es.put("qav",source(d));es.put("phis","phis");es.put("phil","phil");es.put("isx","cd.isx");es.put("isy","cd.isy");es.put("ilx","cd.ilx");es.put("ily","cd.ily");}
   if(d!=3){for(String s:new String[]{"u","v","w1","w2","w3","w4"})es.put(s,s);for(String s:new String[]{"tcs.R_w1","tcs.R_w2","tcs.R_w3"})es.put(s,s);}
   for(Map.Entry<String,String> en:es.entrySet())try{
    m.result().numerical("clProbe").set("expr",new String[]{en.getValue()});double[] val=m.result().numerical("clProbe").getData()[0][0];
    for(int j=0;j<40;j++)p.printf(Locale.US,"%s,%d,%d,%.17g,%.17g,%s,%.17g%n",cid,sol,d,xy[0][j],xy[1][j],en.getKey(),val[j]);
   }catch(Throwable err){System.out.println("POINT_ERROR|"+d+"|"+en.getKey()+"|"+err.getMessage());}
  }p.close();
 }
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);m=ModelUtil.load("ClosureExport",a[0]);data=a[1];String prefix=a[2];
  m.result().numerical().create("clArea","IntSurface");m.result().numerical().create("clLine","IntLine");m.result().numerical().create("clProbe","Interp");
  w=new PrintWriter(prefix+"_integrals.csv","UTF-8");w.println("case,solnum,quadrature_order,kind,domain,boundary,metric,unit,value,status");
  int nb=m.geom("geom1").getNBoundaries();System.out.println("BOUNDARIES|"+nb);
  for(String spec:a[3].split(",")){
   String[] ss=spec.split(":");cid=ss[0];sol=Integer.parseInt(ss[1]);order=8;
   dumpPoints(prefix+"_"+cid+"_points.csv");
   for(int ord:new int[]{4,8}){order=ord;
    for(int d=1;d<=5;d++){
     record("volume",d,0,"area","m2","1");record("volume",d,0,"heat_source","W/m","ht.Q");
     record("volume",d,0,"heat_ohmic","W/m",ohm(d));record("volume",d,0,"heat_activation","W/m",act(d));record("volume",d,0,"heat_reversible","W/m",rev(d));
     record("volume",d,0,"heat_closure_abs","W/m","abs(ht.Q-("+ohm(d)+")-("+act(d)+")-("+rev(d)+"))");
     record("volume",d,0,"storage","W/m","ht.rho*ht.Cp*d(T,t)");
     record("volume",d,0,"advection","W/m",d==1||d==5?"tcs.rho*Cp_g*(u*d(T,x)+v*d(T,y))":"0[W/m^3]");
     record("volume",d,0,"reaction_current","A/m",source(d));
     if(d!=3){record("volume",d,0,"mass_divergence","kg/(m*s)","d(tcs.rho*u,x)+d(tcs.rho*v,y)");record("volume",d,0,"velocity_divergence_abs","m2/s","abs(d(u,x)+d(v,y))");for(String sp:new String[]{"w1","w2","w3"})record("volume",d,0,"species_source_"+sp,"kg/(m*s)","tcs.R_"+sp);}
    }
    for(int b=1;b<=nb;b++){
     double len=integ("IntLine",b,"1"),x=integ("IntLine",b,"x")/len,y=integ("IntLine",b,"y")/len;
     for(int d=1;d<=5;d++){
      double nx=0,ny=0;
      if(Math.abs(x)<1e-10&&y>Y[d-1]&&y<Y[d])nx=-1;
      else if(Math.abs(x-.1)<1e-10&&y>Y[d-1]&&y<Y[d])nx=1;
      else if(x>0&&x<.1&&Math.abs(y-Y[d-1])<1e-10)ny=-1;
      else if(x>0&&x<.1&&Math.abs(y-Y[d])<1e-10)ny=1;
      else continue;
      String flux="("+nx+"*d(T,x)+"+ny+"*d(T,y))";
      record("boundary",d,b,"length","m","1");
      record("boundary",d,b,"heat_out","W/m",trace("-"+(d==1||d==5?"k_g":"k_s")+"*"+flux,d));
      record("boundary",d,b,"heat_out_abs","W/m","abs("+trace("-"+(d==1||d==5?"k_g":"k_s")+"*"+flux,d)+")");
      String recoveredHeat="if(up(dom)=="+d+",ht.ndflux_u,ht.ndflux_d)";
      record("boundary",d,b,"heat_out_recovered","W/m",recoveredHeat);
      record("boundary",d,b,"heat_out_recovered_abs","W/m","abs("+recoveredHeat+")");
      if(d>=2&&d<=4){String ir="if((up(dom)>=2)&&(up(dom)<=4)&&(down(dom)>=2)&&(down(dom)<=4),if(up(dom)=="+d+",-cd.nIl,cd.nIl),cd.nIl)";record("boundary",d,b,"ionic_out_recovered","A/m",ir);record("boundary",d,b,"ionic_out_recovered_abs","A/m","abs("+ir+")");}
      if(d==2||d==4){record("boundary",d,b,"electronic_out_recovered","A/m","cd.nIs");record("boundary",d,b,"electronic_out_recovered_abs","A/m","abs(cd.nIs)");}
      if(d>=2&&d<=4){record("boundary",d,b,"ionic_out","A/m",trace("cd.ilx*"+nx+"+cd.ily*"+ny,d));record("boundary",d,b,"ionic_out_abs","A/m","abs("+trace("cd.ilx*"+nx+"+cd.ily*"+ny,d)+")");}
      if(d==2||d==4){record("boundary",d,b,"electronic_out","A/m",trace("cd.isx*"+nx+"+cd.isy*"+ny,d));record("boundary",d,b,"electronic_out_abs","A/m","abs("+trace("cd.isx*"+nx+"+cd.isy*"+ny,d)+")");}
     }
    }
   }
   System.out.println("EXPORTED|"+cid);System.out.flush();
  }w.close();ModelUtil.remove("ClosureExport");System.out.println("NO_SOLVE_NO_ORIGINAL_SAVE");System.exit(0);
 }
}
