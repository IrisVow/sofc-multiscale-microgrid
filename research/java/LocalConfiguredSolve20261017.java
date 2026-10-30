import com.comsol.model.*;import com.comsol.model.util.*;
public class LocalConfiguredSolve20261017{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("LocalConfigured",a[0]);System.out.println("LOCAL_SOLVE_START|vertices="+m.mesh("mesh1").getVertex()[0].length);System.out.flush();m.sol("sol1").runAll();m.save(a[1]);m.save(a[1].replace(".mph","_history.java"),"java");System.out.println("LOCAL_SOLVE_COMPLETE");System.exit(0);}catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
