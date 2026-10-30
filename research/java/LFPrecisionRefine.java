import com.comsol.model.*;import com.comsol.model.util.*;
public class LFPrecisionRefine{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("LFPrecision",a[0]);m.sol("sol1").copySolution("lfPrecisionWarm");m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","lfPrecisionWarm");m.sol("sol1").feature("v1").set("solnum","1");m.sol("sol1").feature("s1").set("stol","1e-8");m.sol("sol1").feature("s1").feature("fc1").set("initstep","1");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","15");
  for(String tag:m.sol("sol1").feature("s1").feature().tags())if(m.sol("sol1").feature("s1").feature(tag).getType().equals("Direct")){m.sol("sol1").feature("s1").feature(tag).set("nliniterrefine","on");m.sol("sol1").feature("s1").feature(tag).set("iterrefine","on");m.sol("sol1").feature("s1").feature(tag).set("maxrefinesteps",25);System.out.println("ITERATIVE_REFINEMENT|"+tag);}
  m.sol("sol1").runAll();m.label("Conservative physical steady state; precision-refined warm solve, tolerance1e-8");m.save(a[1]);m.save(a[1].replace(".mph","_history.java"),"java");System.out.println("PRECISION_REFINE_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
