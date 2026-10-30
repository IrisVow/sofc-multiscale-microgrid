import com.comsol.model.*;import com.comsol.model.util.*;
public class ReactiveSolveContinuation{
 static Model m;static String prefix;static int seq=0;
 static void solve(String label,double v,double alpha)throws Exception{
  m.param().set("V_cell",v+"[V]");m.param().set("alpha_rxn",Double.toString(alpha));m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{Double.toString(v)}});
  System.out.println("SOLVE_START|"+label+"|V="+v+"|alpha="+alpha);System.out.flush();m.sol("sol1").runAll();
  String path=prefix+"_state"+seq+".mph";m.save(path);System.out.println("SOLVE_SAVED|"+label+"|V="+v+"|alpha="+alpha+"|"+path);System.out.flush();
  String tag="contWarm"+seq++;m.sol("sol1").copySolution(tag);m.sol("sol1").feature("v1").set("initsol",tag);m.sol("sol1").feature("v1").set("solnum","1");
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);m=ModelUtil.load("ReactiveContinuation",a[0]);prefix=a[1];
  m.sol("sol1").feature("v1").set("initsol","parentWarm");m.sol("sol1").feature("v1").set("solnum","1");
  m.sol("sol1").feature("s1").feature("fc1").set("initstep","0.5");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","100");
  for(double v:new double[]{1.,.95,.9,.85,.8})solve("voltage_initialization",v,0.);
  for(double alpha:new double[]{.25,.5,.75,1.})solve("S5_source_continuation",.8,alpha);
  m.save(prefix+"_S5.mph");System.out.println("FULL_REACTIVE_S5_SAVED");System.out.flush();
  solve("S6_full_source",.75,1.);m.save(prefix+"_S6.mph");System.out.println("FULL_REACTIVE_S6_SAVED");System.out.flush();ModelUtil.remove("ReactiveContinuation");System.exit(0);
 }catch(Throwable e){System.out.println("CONTINUATION_SOLVE_FAILED");e.printStackTrace(System.out);System.out.flush();System.exit(1);}}
}
