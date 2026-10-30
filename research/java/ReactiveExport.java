import com.comsol.model.*;import com.comsol.model.util.*;import com.comsol.model.physics.*;import java.util.*;import java.io.*;
public class ReactiveExport extends ClosureExport{
 static boolean speciesFlux=true;
 public static void main(String[] a){try{run(a);}catch(Throwable e){System.out.println("EXPORT_FAILED");e.printStackTrace(System.out);System.out.flush();System.exit(1);}}
 static void properties(PhysicsFeature f,String path,PrintWriter p){
  p.println("FEATURE|"+path+"|"+f.getType()+"|active="+f.isActive()+"|domains="+Arrays.toString(f.selection().entities()));
  for(String key:f.properties())try{p.println(path+"|"+key+"|"+Arrays.deepToString(f.getStringMatrix(key)));}catch(Throwable e){try{p.println(path+"|"+key+"|"+Arrays.toString(f.getStringArray(key)));}catch(Throwable ee){}}
  for(String tag:f.feature().tags())properties(f.feature(tag),path+"/"+tag,p);
 }
 static String spTrace(String e,int dom,int boundary){
  // At gas/electrode interfaces both species domains are active; elsewhere
  // the built-in normal flux is already outward from the sole active side.
  if(boundary==4||boundary==10)return "if(up(dom)=="+dom+",-("+e+"),("+e+"))";
  return e;
 }
 static void summary(String prefix)throws Exception{
  PrintWriter p=new PrintWriter(prefix+"_summary.csv","UTF-8");p.println("case,metric,value");
  for(String name:new String[]{"V_cell","T_in","FU","lambda_air","i_avg","eps_p","tau_p","alpha_rxn","u_in_a","u_in_c"})p.printf(Locale.US,"%s,%s,%.17g%n",cid,name,m.param().evaluate(name));
  p.printf(Locale.US,"%s,mesh_vertices,%d%n",cid,m.mesh("mesh1").getVertex()[0].length);
  for(String type:new String[]{"MaxSurface","MinSurface"}){
   String tag="rxSummary"+type;m.result().numerical().create(tag,type);m.result().numerical(tag).set("data",data);m.result().numerical(tag).set("solnum",1);
   for(String e:new String[]{"T","w1","w2","w3","w4","w1+w2+w3+w4","tcs.rho","tcs.pA","fp.rho-tcs.rho","tcs.u-u","tcs.v-v"}){
    m.result().numerical(tag).selection().set(e.equals("T")?new int[]{1,2,3,4,5}:new int[]{1,2,4,5});m.result().numerical(tag).set("expr",new String[]{e});
    p.printf(Locale.US,"%s,%s_%s,%.17g%n",cid,type.startsWith("Max")?"max":"min",e.replace('+','_').replace('.','_'),m.result().numerical(tag).getReal()[0][0]);
   }
  }p.close();
  PrintWriter settings=new PrintWriter(prefix+"_settings.txt","UTF-8");
  for(String ph:new String[]{"tcs","fp","ht"}){
   settings.println("PHYSICS|"+ph);
   for(String tag:m.physics(ph).feature().tags())properties(m.physics(ph).feature(tag),ph+"/"+tag,settings);
   for(String tag:m.physics(ph).prop().tags())for(String key:m.physics(ph).prop(tag).properties())try{settings.println("PROP|"+ph+"|"+tag+"|"+key+"|"+Arrays.toString(m.physics(ph).prop(tag).getStringArray(key)));}catch(Throwable e){}
  }
  for(String tag:m.component("comp1").multiphysics().tags()){
   settings.println("COUPLING|"+tag+"|"+m.component("comp1").multiphysics(tag).getType());
   for(String key:m.component("comp1").multiphysics(tag).properties())try{settings.println("COUPLING_PROP|"+tag+"|"+key+"|"+Arrays.toString(m.component("comp1").multiphysics(tag).getStringArray(key)));}catch(Throwable e){}
  }settings.close();
 }
 static void probeFluxNames(String prefix)throws Exception{
  PrintWriter p=new PrintWriter(prefix+"_variable_probe.txt","UTF-8");
  m.result().numerical("clLine").set("data",data);m.result().numerical("clLine").set("solnum",1);m.result().numerical("clLine").selection().set(new int[]{9});
  for(String e:new String[]{"tcs.ntflux_w1","tcs.ndflux_w1","tcs.ntflux_w2","tcs.ntflux_w3","tcs.ntflux_w4","tcs.ndflux_w4","tcs.rho*(u*nx+v*ny)","fp.nmflux","fp.nflux"})try{
   m.result().numerical("clLine").set("expr",new String[]{e});p.println("OK|"+e+"|"+m.result().numerical("clLine").getReal()[0][0]);
  }catch(Throwable err){p.println("ERROR|"+e+"|"+err.getMessage());if(e.equals("tcs.ntflux_w1"))speciesFlux=false;}
  p.close();if(!speciesFlux)throw new RuntimeException("Native species flux name unresolved; inspect variable probe.");
 }
 static void gasProfiles(String prefix)throws Exception{
  PrintWriter p=new PrintWriter(prefix+"_profiles.csv","UTF-8");p.println("case,domain,x,y,metric,value");
  for(int d:new int[]{1,5}){
   int n=201;double[][] xy=new double[2][n];for(int j=0;j<n;j++){xy[0][j]=.1*j/(n-1);xy[1][j]=(Y[d-1]+Y[d])/2;}
   m.result().numerical("clProbe").set("data",data);m.result().numerical("clProbe").set("solnum",1);m.result().numerical("clProbe").set("coord",xy);
   for(String e:new String[]{"T","u","v","p","w1","w2","w3","w4","tcs.rho"}){
    m.result().numerical("clProbe").set("expr",new String[]{e});double[] val=m.result().numerical("clProbe").getData()[0][0];
    for(int j=0;j<n;j++)p.printf(Locale.US,"%s,%d,%.17g,%.17g,%s,%.17g%n",cid,d,xy[0][j],xy[1][j],e.replace('.','_'),val[j]);
   }
  }p.close();
 }
 static void run(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showProgress(false);ModelUtil.showPlots(false);m=ModelUtil.load("ReactiveExport",a[0]);data="dset1";sol=1;cid=a[2];String prefix=a[1];
  if(Math.abs(m.param().evaluate("alpha_rxn")-1)>1e-12)throw new RuntimeException("Not final full-source state");
  if(!m.physics("tcs").feature("pom1").isActive())throw new RuntimeException("Porous species disabled");
  m.result().numerical().create("clArea","IntSurface");m.result().numerical().create("clLine","IntLine");m.result().numerical().create("clProbe","Interp");
  summary(prefix);probeFluxNames(prefix);order=8;dumpPoints(prefix+"_"+cid+"_points.csv");gasProfiles(prefix);
  w=new PrintWriter(prefix+"_integrals.csv","UTF-8");w.println("case,solnum,quadrature_order,kind,domain,boundary,metric,unit,value,status");
  int nb=m.geom("geom1").getNBoundaries();
  for(int ord:new int[]{4,8}){order=ord;
   for(int d=1;d<=5;d++){
    record("volume",d,0,"area","m2","1");record("volume",d,0,"heat_source","W/m","ht.Q");record("volume",d,0,"heat_ohmic","W/m",ohm(d));record("volume",d,0,"heat_activation","W/m",act(d));record("volume",d,0,"heat_reversible","W/m",rev(d));
    record("volume",d,0,"heat_closure_abs","W/m","abs(ht.Q-("+ohm(d)+")-("+act(d)+")-("+rev(d)+"))");record("volume",d,0,"storage","W/m","0[W/m^3]");
    record("volume",d,0,"advection","W/m",d==1||d==5?"tcs.rho*Cp_g*(u*d(T,x)+v*d(T,y))":"0[W/m^3]");
    record("volume",d,0,"reaction_current","A/m",source(d));
    if(d!=3){
     for(String sp:new String[]{"w1","w2","w3","w4"})record("volume",d,0,"species_source_"+sp,"kg/(m*s)","tcs.R_"+sp);
     record("volume",d,0,"gas_mass_source","kg/(m*s)","tcs.R_w1+tcs.R_w2+tcs.R_w3+tcs.R_w4");
     record("volume",d,0,"mass_divergence","kg/(m*s)","d(tcs.rho*u,x)+d(tcs.rho*v,y)");
     String expected=d==2?"cd.iloc_pcar1*Av_ca*M_O2/(4*F_const)":d==4?"cd.iloc_panr1*Av_an*(M_H2O-M_H2)/(2*F_const)":"0[kg/(m^3*s)]";
     record("volume",d,0,"expected_gas_mass_source","kg/(m*s)",expected);
     record("volume",d,0,"source_identity_abs","kg/(m*s)","abs(tcs.R_w1+tcs.R_w2+tcs.R_w3+tcs.R_w4-("+expected+"))");
    }
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
     String vel="(u*"+nx+"+v*"+ny+")",rawheat=trace("-"+(d==1||d==5?"k_g":"k_s")+"*(d(T,x)*"+nx+"+d(T,y)*"+ny+")",d);
     String rec="if(up(dom)=="+d+",ht.ndflux_u,ht.ndflux_d)";
     record("boundary",d,b,"length","m","1");record("boundary",d,b,"heat_out","W/m",rawheat);record("boundary",d,b,"heat_out_abs","W/m","abs("+rawheat+")");record("boundary",d,b,"heat_out_recovered","W/m",rec);record("boundary",d,b,"heat_out_recovered_abs","W/m","abs("+rec+")");
     if(d>=2&&d<=4){String ir="if((up(dom)>=2)&&(up(dom)<=4)&&(down(dom)>=2)&&(down(dom)<=4),if(up(dom)=="+d+",-cd.nIl,cd.nIl),cd.nIl)";record("boundary",d,b,"ionic_out_recovered","A/m",ir);record("boundary",d,b,"ionic_out_recovered_abs","A/m","abs("+ir+")");}
     if(d==2||d==4){record("boundary",d,b,"electronic_out_recovered","A/m","cd.nIs");record("boundary",d,b,"electronic_out_recovered_abs","A/m","abs(cd.nIs)");}
     if(d!=3){
      String flow=trace("tcs.rho*"+vel,d);record("boundary",d,b,"mass_out","kg/(m*s)",flow);record("boundary",d,b,"mass_out_abs","kg/(m*s)","abs("+flow+")");
      for(String sp:new String[]{"w1","w2","w3","w4"}){
       String conv=trace("tcs.rho*"+sp+"*"+vel,d),tot=spTrace("tcs.ntflux_"+sp,d,b);
       record("boundary",d,b,"species_convective_out_"+sp,"kg/(m*s)",conv);
       record("boundary",d,b,"species_out_"+sp,"kg/(m*s)",tot);record("boundary",d,b,"species_out_abs_"+sp,"kg/(m*s)","abs("+tot+")");
      }
     }
    }
   }
  }
  w.close();ModelUtil.remove("ReactiveExport");System.out.println("REACTIVE_EXPORT_COMPLETE|"+cid);System.out.flush();System.exit(0);
 }
}
