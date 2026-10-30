import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;import java.io.*;
public class LFAnnotateV3 extends ClosureExport{
 static String outward(String v){return "if(up(dom)>0,-uflux_spatial("+v+"),-dflux_spatial("+v+"))";}
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);m=ModelUtil.load("LFAnnotated",a[0]);String before=LFFrozenRecovery.hash(m.sol("sol1").getU());String[] names={"mass","H2","H2O","O2","N2"};
  m.component("comp1").variable().create("lfInterfaceFlux");m.component("comp1").variable("lfInterfaceFlux").selection().geom("geom1",1);m.component("comp1").variable("lfInterfaceFlux").selection().set(new int[]{4,10});
  m.component("comp1").variable().create("lfPortFlux");m.component("comp1").variable("lfPortFlux").selection().geom("geom1",1);m.component("comp1").variable("lfPortFlux").selection().set(new int[]{1,9,12,16});
  String raw="withsol('sol1',fp.rho*(u*nx+v*ny))",mr=outward("lfR");
  for(int i=0;i<5;i++){
   for(String side:new String[]{"up","down"}){String ff=i==4?"lf_num_mass_"+side+"-lf_num_H2_"+side+"-lf_num_H2O_"+side+"-lf_num_O2_"+side:"-"+(side.equals("up")?"uflux_spatial(":"dflux_spatial(")+(i==0?"lfR":"lfSR"+i)+")";m.component("comp1").variable("lfInterfaceFlux").set("lf_num_"+names[i]+"_"+side,ff,"Numerical outward flux from "+side+" side; N2 derived, kg/(m^2*s)");}
   String pp=i==0?raw:i==4?"lf_num_mass_port-lf_num_H2_port-lf_num_H2O_port-lf_num_O2_port":"if(x<0.05[m],("+outward("lfSR"+i)+")+withsol('sol1',w"+i+")*(("+raw+")-("+mr+")),withsol('sol1',fp.rho*w"+i+"*(u*nx+v*ny)))";
   m.component("comp1").variable("lfPortFlux").set("lf_num_"+names[i]+"_port",pp,"Conservative port flux; inlet advective moments preserved, kg/(m^2*s)");
  }
  m.study("std1").label("Physical steady state: conservative weak forms, quadrature8");m.study("lfStudy").label("Flux recovery: frozen physical solution");m.physics("tcs").label("Species: conservative weak form; numerical flux uses lf_num_ variables");m.result().dataset("lfData").label("Numerical fluxes: lf_num_ variables; physical solution frozen");
  for(int b:new int[]{4,10}){String tag="lfFlux"+b;m.result().create(tag,"PlotGroup1D");m.result(tag).label("Conservative mass flux, interface "+b);m.result(tag).set("data","lfData");m.result(tag).create("line1","LineGraph");m.result(tag).feature("line1").selection().set(new int[]{b});m.result(tag).feature("line1").set("expr","lf_num_mass_up");m.result(tag).feature("line1").set("xdata","expr");m.result(tag).feature("line1").set("xdataexpr","x");}
  m.sol(m.result().dataset("lfData").getString("solution")).updateSolution();data="lfData";sol=1;order=8;m.result().numerical().create("lfNamedCheck","IntLine");m.result().numerical("lfNamedCheck").set("data",data);m.result().numerical("lfNamedCheck").set("intorder",8);PrintWriter p=new PrintWriter(a[2],"UTF-8");p.println("boundary,quantity,value");
  for(int b:new int[]{1,9,12,16})for(String name:names){m.result().numerical("lfNamedCheck").selection().set(new int[]{b});m.result().numerical("lfNamedCheck").set("expr",new String[]{"lf_num_"+name+"_port"});p.printf(Locale.US,"%d,%s,%.17g%n",b,name,m.result().numerical("lfNamedCheck").getReal()[0][0]);}p.close();
  if(!before.equals(LFFrozenRecovery.hash(m.sol("sol1").getU())))throw new RuntimeException("Annotation changed physical data");m.label("Conservative interface flux review model; see lf_num_ variables and validation report");m.save(a[1]);System.out.println("ANNOTATED_REVIEW_MODEL_SAVED");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
