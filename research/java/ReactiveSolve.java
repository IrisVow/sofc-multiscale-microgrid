import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;
public class ReactiveSolve{
 public static void main(String[] a){try{run(a);}catch(Throwable e){System.out.println("SOLVE_FAILED");e.printStackTrace(System.out);System.out.flush();System.exit(1);}}
 static void run(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("ReactiveSolve",a[0]);
  String prefix=a[1];
  double[] ramps={0.,.25,.5,.75,1.};
  for(int k=0;k<ramps.length;k++){
   double alpha=ramps[k];m.param().set("alpha_rxn",Double.toString(alpha));System.out.println("SOLVE_START|S5|alpha="+alpha);System.out.flush();
   m.sol("sol1").runAll();String state=prefix+"_S5_alpha"+k+".mph";m.save(state);System.out.println("SOLVE_SAVED|S5|alpha="+alpha+"|"+state);System.out.flush();
   if(k<ramps.length-1){String warm="warmAlpha"+k;m.sol("sol1").copySolution(warm);m.sol("sol1").feature("v1").set("initsol",warm);m.sol("sol1").feature("v1").set("solnum","1");}
  }
  m.save(prefix+"_S5.mph");
  m.sol("sol1").copySolution("warmS5");m.sol("sol1").feature("v1").set("initsol","warmS5");m.sol("sol1").feature("v1").set("solnum","1");
  m.param().set("V_cell","0.75[V]");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{"0.75"}});
  System.out.println("SOLVE_START|S6|alpha=1.0");System.out.flush();m.sol("sol1").runAll();m.save(prefix+"_S6.mph");System.out.println("SOLVE_SAVED|S6|alpha=1.0");System.out.flush();
  ModelUtil.remove("ReactiveSolve");System.exit(0);
 }
}
