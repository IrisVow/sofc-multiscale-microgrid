import com.comsol.model.*;import com.comsol.model.util.*;
public class ReactiveSolveExplicitMass{
 static Model m;static String prefix;static int seq=0;
 static void solve(double v)throws Exception{
  m.param().set("V_cell",v+"[V]");m.param().set("alpha_rxn","1");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{Double.toString(v)}});
  System.out.println("SOLVE_START|V="+v+"|alpha=1");System.out.flush();m.sol("sol1").runAll();
  m.save(prefix+"_latest.mph");System.out.println("SOLVE_SAVED|V="+v+"|alpha=1");System.out.flush();
  String tag="fineWarm"+seq++;m.sol("sol1").copySolution(tag);m.sol("sol1").feature("v1").set("initsol",tag);m.sol("sol1").feature("v1").set("solnum","1");
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);m=ModelUtil.load("ReactiveFine",a[0]);prefix=a[1];
  m.component("comp1").multiphysics().remove("rxnflow");m.physics("fp").create("massFaraday","MassSource",2);m.physics("fp").feature("massFaraday").selection().set(new int[]{2,4});m.physics("fp").feature("massFaraday").set("Qm","tcs.R_w1+tcs.R_w2+tcs.R_w3+tcs.R_w4");m.physics("fp").feature("massFaraday").label("Net Faradaic gas mass: source once, per total electrode volume");m.save(prefix+"_configured.mph");
  m.sol("sol1").copySolution("fineInitial");m.sol("sol1").feature("v1").set("initsol","fineInitial");m.sol("sol1").feature("v1").set("solnum","1");
  m.sol("sol1").feature("s1").feature("fc1").set("initstep","0.1");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","100");
  for(int k=0;k<=25;k++){double v=(100-k)/100.;solve(v);if(k==20){m.save(prefix+"_S5.mph");System.out.println("FULL_REACTIVE_S5_SAVED");}if(k==25){m.save(prefix+"_S6.mph");System.out.println("FULL_REACTIVE_S6_SAVED");}System.out.flush();}
  System.exit(0);
 }catch(Throwable e){System.out.println("FINE_SOLVE_FAILED");e.printStackTrace(System.out);System.out.flush();System.exit(1);}}
}
