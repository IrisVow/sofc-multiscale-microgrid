import com.comsol.model.*;import com.comsol.model.util.*;
public class ReactiveSolveSegregatedV4{
 static Model m;static String prefix;static int seq=0;
 static void solve(double v)throws Exception{m.param().set("V_cell",v+"[V]");m.param().set("alpha_rxn","1");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{Double.toString(v)}});System.out.println("SOLVE_START|V="+v+"|alpha=1");System.out.flush();m.sol("sol1").runAll();m.save(prefix+"_latest.mph");System.out.println("SOLVE_SAVED|V="+v+"|alpha=1");System.out.flush();String tag="segWarm"+seq++;m.sol("sol1").copySolution(tag);m.sol("sol1").feature("v1").set("initsol",tag);m.sol("sol1").feature("v1").set("solnum","1");}
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);m=ModelUtil.load("ReactiveSeg",a[0]);prefix=a[1];m.sol("sol1").copySolution("segInitial");m.sol("sol1").feature("v1").set("initsol","segInitial");m.sol("sol1").feature("v1").set("solnum","1");m.sol("sol1").feature("s1").create("segReactive","Segregated");m.sol("sol1").feature("s1").feature().remove("fc1");m.sol("sol1").feature("s1").feature("segReactive").set("maxsegiter",100);
 for(String tag:m.sol("sol1").feature("s1").feature("segReactive").feature().tags())m.sol("sol1").feature("s1").feature("segReactive").feature().remove(tag);
 m.sol("sol1").feature("s1").create("dReactive","Direct");
 String[][] groups={{"comp1_phis","comp1_phil"},{"comp1_w1","comp1_w2","comp1_w3","comp1_u","comp1_p"},{"comp1_T"}};
 for(int k=0;k<groups.length;k++){String tag="group"+k;m.sol("sol1").feature("s1").feature("segReactive").create(tag,"SegregatedStep");m.sol("sol1").feature("s1").feature("segReactive").feature(tag).set("segvar",groups[k]);m.sol("sol1").feature("s1").feature("segReactive").feature(tag).set("subdamp","0.5");m.sol("sol1").feature("s1").feature("segReactive").feature(tag).set("linsolver","dReactive");}
 for(int k=0;k<=25;k++){double v=(100-k)/100.;solve(v);if(k==20){m.save(prefix+"_S5.mph");System.out.println("FULL_REACTIVE_S5_SAVED");}if(k==25){m.save(prefix+"_S6.mph");System.out.println("FULL_REACTIVE_S6_SAVED");}System.out.flush();}System.exit(0);
 }catch(Throwable e){System.out.println("SEG_SOLVE_FAILED");e.printStackTrace(System.out);System.out.flush();System.exit(1);}}
}
