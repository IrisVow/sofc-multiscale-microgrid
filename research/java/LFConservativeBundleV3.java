import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;import java.io.*;
public class LFConservativeBundleV3 extends ClosureExport{
 static String old(String e){return "withsol('sol1',("+e+"))";}
 static String field(int i){return i==0?"lfR":"lfSR"+i;}
 static String rec(int i,int d){return "if(up(dom)=="+d+",-uflux_spatial("+field(i)+"),-dflux_spatial("+field(i)+"))";}
 static boolean port(int b){return b==1||b==9||b==12||b==16;}
 static String previous(int i,int d,int b,double nx,double ny){if(i==0)return trace(old("fp.rho*(u*"+nx+"+v*"+ny+")"),d);return old(FluxAuditExport.sidedDiff("w"+i,d));}
 static String flux(int i,int d,int b,double nx,double ny){if(port(b)){if(i==0)return previous(0,d,b,nx,ny);if(b==1||b==9)return rec(i,d);return old(trace("fp.rho*w"+i+"*(u*"+nx+"+v*"+ny+")",d));}if(b!=4&&b!=10)return "0[kg/(m^2*s)]";return rec(i,d);}
 static void configure(){
  for(int i=0;i<4;i++){String tag=i==0?"lfMass":"lfSpecies"+i,variable=field(i),sp="w"+i;
   m.component("comp1").physics().create(tag,"GeneralFormPDE","geom1");m.physics(tag).selection().set(new int[]{1,2,4,5});m.physics(tag).field("dimensionless").field(variable);m.physics(tag).field("dimensionless").component(new String[]{variable});
   m.physics(tag).prop("Units").set("DependentVariableQuantity","none");m.physics(tag).prop("Units").set("CustomDependentVariableUnit","kg/m^3");m.physics(tag).prop("Units").set("SourceTermQuantity","none");m.physics(tag).prop("Units").set("CustomSourceTermUnit","kg/(m^3*s)");
   m.physics(tag).prop("ShapeProperty").set("order",1);m.physics(tag).prop("ShapeProperty").set("boundaryFlux",true);m.physics(tag).prop("ShapeProperty").set("boundaryFluxSmooth",false);
   String gx=i==0?"fp.rho*u":"tcs.j_"+sp+"x+fp.rho*u*"+sp,gy=i==0?"fp.rho*v":"tcs.j_"+sp+"y+fp.rho*v*"+sp,src=i==0?"fp.Qm":"tcs.R_"+sp;
   m.physics(tag).feature("gfeq1").set("Ga",new String[]{old(gx),old(gy)});m.physics(tag).feature("gfeq1").set("f",old(src)+"-"+variable+"/1[s]");m.physics(tag).feature("gfeq1").set("da","0");m.physics(tag).feature("gfeq1").set("ea","0");
   m.physics(tag).create("lfPorts","FluxBoundary",1);m.physics(tag).feature("lfPorts").selection().set(i==0?new int[]{1,9,12,16}:new int[]{12,16});m.physics(tag).feature("lfPorts").set("g","-("+old(i==0?"fp.rho*(u*nx+v*ny)":"fp.rho*"+sp+"*(u*nx+v*ny)")+")");m.physics(tag).feature("lfPorts").set("q","0");
   if(i>0){m.physics(tag).create("lfInlet","DirichletBoundary",1);m.physics(tag).feature("lfInlet").selection().set(new int[]{1,9});m.physics(tag).feature("lfInlet").set("r","0");}
   for(String[] row:m.physics(tag).feature("gfeq1").featureInfo("info").getInfoTable("Weak"))m.physics(tag).feature("gfeq1").featureInfo("info").set(row[2],new String[]{"0"});
   LFPrimaryConservative.weak(m,"lfAux"+i,2,new int[]{1,2,4,5},"("+old(gx)+")*test("+variable+"x)+("+old(gy)+")*test("+variable+"y)+("+old(src)+"-"+variable+"/1[s])*test("+variable+")");
  }
  m.study().create("lfStudy");m.study("lfStudy").create("stat","Stationary");ArrayList<String> active=new ArrayList<>(),primary=new ArrayList<>();for(String tag:m.physics().tags()){active.add(tag);active.add(tag.startsWith("lf")?"on":"off");primary.add(tag);primary.add(tag.startsWith("lf")?"off":"on");}
  m.study("lfStudy").feature("stat").set("activate",active.toArray(new String[0]));m.study("lfStudy").feature("stat").set("useadvanceddisable","on");m.study("lfStudy").feature("stat").set("disabledcommon",new String[]{"mcVolume","mcWall","lfCont","lfContPorts","lfSpecies","lfSpeciesPorts"});m.study("std1").feature("stat").set("activate",primary.toArray(new String[0]));m.study("std1").feature("stat").set("useadvanceddisable","on");m.study("std1").feature("stat").set("disabledcommon",new String[]{"lfAux0","lfAux1","lfAux2","lfAux3"});m.study("lfStudy").createAutoSequences("all");
 }
 static void export(String prefix)throws Exception{
  m.result().dataset().create("lfData","Solution");String aux="";for(String tag:m.sol().tags())if(m.sol(tag).study().equals("lfStudy"))aux=tag;m.result().dataset("lfData").set("solution",aux);data="lfData";sol=1;
  for(String tag:new String[]{"clLine","clArea"})if(Arrays.asList(m.result().numerical().tags()).contains(tag))m.result().numerical().remove(tag);m.result().numerical().create("clLine","IntLine");m.result().numerical().create("clArea","IntSurface");
  w=new PrintWriter(prefix+"_integrals.csv","UTF-8");w.println("case,solnum,quadrature_order,kind,domain,boundary,metric,unit,value,status");
  for(int ord:new int[]{4,8}){order=ord;
   for(int d:new int[]{1,2,4,5})for(int i=0;i<4;i++){String name=i==0?"mass":"w"+i;record("volume",d,0,"source_"+name,"kg/(m*s)",old(i==0?"fp.Qm":"tcs.R_w"+i));record("volume",d,0,"lifting_"+name,"kg/(m*s)",field(i)+"/1[s]");record("volume",d,0,"lifting_abs_"+name,"kg/(m*s)","abs("+field(i)+"/1[s])");}
   for(int b=1;b<=m.geom("geom1").getNBoundaries();b++){double len=integ("IntLine",b,"1"),xx=integ("IntLine",b,"x")/len,yy=integ("IntLine",b,"y")/len;
    for(int d:new int[]{1,2,4,5}){double nx=0,ny=0;if(Math.abs(xx)<1e-10&&yy>Y[d-1]&&yy<Y[d])nx=-1;else if(Math.abs(xx-.1)<1e-10&&yy>Y[d-1]&&yy<Y[d])nx=1;else if(xx>0&&xx<.1&&Math.abs(yy-Y[d-1])<1e-10)ny=-1;else if(xx>0&&xx<.1&&Math.abs(yy-Y[d])<1e-10)ny=1;else continue;
     String n2="("+flux(0,d,b,nx,ny)+")-("+flux(1,d,b,nx,ny)+")-("+flux(2,d,b,nx,ny)+")-("+flux(3,d,b,nx,ny)+")";
     for(int i=0;i<5;i++){String name=i==0?"mass":"w"+i,ff=i==4?n2:flux(i,d,b,nx,ny);record("boundary",d,b,"final_"+name,"kg/(m*s)",ff);record("boundary",d,b,"final_abs_"+name,"kg/(m*s)","abs("+ff+")");if(i<4){record("boundary",d,b,"raw_recovered_"+name,"kg/(m*s)",rec(i,d));record("boundary",d,b,"previous_"+name,"kg/(m*s)",previous(i,d,b,nx,ny));}}
    }
   }
  }w.close();PrintWriter p=new PrintWriter(prefix+"_interface_L1.csv","UTF-8");p.println("case,quadrature_order,boundary,quantity,signed_jump,L1_jump,L1_flux_scale,N2_flux_L1_relative_to_mass");
  for(int ord:new int[]{4,8}){order=ord;for(int b:new int[]{4,10}){int da=b==4?1:4,db=b==4?2:5;String n2a="("+rec(0,da)+")-("+rec(1,da)+")-("+rec(2,da)+")-("+rec(3,da)+")",n2b="("+rec(0,db)+")-("+rec(1,db)+")-("+rec(2,db)+")-("+rec(3,db)+")";double mn=integ("IntLine",b,"abs("+rec(0,da)+")+abs("+rec(0,db)+")"),n2norm=integ("IntLine",b,"abs("+n2a+")+abs("+n2b+")");for(int i=0;i<5;i++){String fa=i==4?n2a:rec(i,da),fb=i==4?n2b:rec(i,db),jump="("+fa+")+("+fb+")";p.printf(Locale.US,"%s,%d,%d,%s,%.17g,%.17g,%.17g,%.17g%n",cid,ord,b,i==0?"mass":"w"+i,integ("IntLine",b,jump),integ("IntLine",b,"abs("+jump+")"),integ("IntLine",b,"abs("+fa+")+abs("+fb+")"),n2norm/Math.max(mn,1e-12));p.flush();}}}p.close();
 }
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[3]);m=ModelUtil.load("LFBundle",a[0]);cid=a[2];String before=LFFrozenRecovery.hash(m.sol("sol1").getU());configure();String aux="";for(String tag:m.sol().tags())if(m.sol(tag).study().equals("lfStudy"))aux=tag;m.sol(aux).feature("s1").set("stol","1e-10");System.out.println("BUNDLE_START|"+cid);System.out.flush();m.sol(aux).runAll();String after=LFFrozenRecovery.hash(m.sol("sol1").getU());if(!before.equals(after))throw new RuntimeException("Frozen solution changed");export(a[1]);PrintWriter p=new PrintWriter(a[1]+"_metadata.json","UTF-8");p.println("{\"original_solution_sha256\":\""+before+"\",\"frozen_solution_unchanged\":true,\"auxiliary_solver\":\""+aux+"\",\"method\":\"four common-space conservative passive flux recoveries\"}");p.close();m.label("Common-space conservative flux recovery; physical fields frozen; validation required");m.save(a[1]+".mph");m.save(a[1]+"_history.java","java");System.out.println("BUNDLE_COMPLETE|"+cid);System.exit(0);}catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
