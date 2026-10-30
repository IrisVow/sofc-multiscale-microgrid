import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
/** Per-trajectory stationary initial condition at the actual waveform value at t=0. */
public class ReferenceInitialSolve20261017{
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("ReferenceInitial",a[0]);ReferenceSteadySolve20261017.freezeLocalMesh(m);String mesh=ReferenceSteadySolve20261017.meshHash(m);
  m.sol("sol1").copySolution("referenceInitialWarm");m.param().set("V_cell",a[3]+"[V]");m.param().set("T_in",a[4]+"[K]");m.param().set("lambda_air",a[5]);m.param().set("FU",a[6]);m.param().set("V_step_amp",a[7]+"[V]");
  if(a[8].equals("legacy_step")){
   if(!m.func("step_load").getType().equals("Step"))throw new RuntimeException("Expected original Step function");m.func("step_load").set("location",a[9]+"[s]");m.func("step_load").set("smooth",.1);m.func("step_load").set("smoothactive",true);
  }else{
   m.func().remove("step_load");m.func().create("step_load","Analytic");m.func("step_load").set("funcname","step_load");m.func("step_load").set("args",new String[]{"t"});m.func("step_load").set("argunit",new String[]{"s"});m.func("step_load").set("fununit","1");m.func("step_load").set("expr",a[8].replaceAll("\\bt\\b","(t/1[s])"));
  }
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","referenceInitialWarm");m.sol("sol1").feature("v1").set("solnum","1");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{a[3]}});
  System.out.println("INITIAL_STEADY_START|wave="+a[8]);System.out.flush();m.sol("sol1").runAll();if(!ReferenceSteadySolve20261017.meshHash(m).equals(mesh))throw new RuntimeException("Initial-condition mesh changed");
  m.label("Per-trajectory steady initial condition; acceptance pending");m.save(a[1]);System.out.println("INITIAL_STEADY_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
