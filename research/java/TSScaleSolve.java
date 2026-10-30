import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class TSScaleSolve {
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("TSScale",a[0]);m.sol("sol1").copySolution("tsWarm");
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","tsWarm");m.sol("sol1").feature("v1").set("solnum","1");
  for(String w:new String[]{"w1","w2","w3"}){m.sol("sol1").feature("v1").feature("comp1_"+w).set("scalemethod","manual");m.sol("sol1").feature("v1").feature("comp1_"+w).set("scaleval",a[3]);}
  m.sol("sol1").feature("s1").set("stol",a[4]);m.sol("sol1").feature("s1").feature("fc1").set("dtech","const");m.sol("sol1").feature("s1").feature("fc1").set("damp","1");m.sol("sol1").feature("s1").feature("fc1").set("jtech","onevery");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","12");m.sol("sol1").feature("s1").feature("fc1").set("termonres","off");
  for(String tag:m.sol("sol1").feature("s1").feature().tags())if(m.sol("sol1").feature("s1").feature(tag).getType().equals("Direct")){m.sol("sol1").feature("s1").feature(tag).set("nliniterrefine","on");m.sol("sol1").feature("s1").feature(tag).set("iterrefine","on");m.sol("sol1").feature("s1").feature(tag).set("maxrefinesteps",25);}
  System.out.println("SCALED_CONSTANT_NEWTON_START|scale="+a[3]+"|tol="+a[4]);System.out.flush();m.sol("sol1").runAll();m.label("Trace species scale and constant Newton precision branch; acceptance pending");m.save(a[1]);m.save(a[1].replace(".mph","_history.java"),"java");System.out.println("TS_SCALED_SOLVE_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
