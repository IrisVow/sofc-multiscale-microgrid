import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
/** Reuse an accepted identical steady field, or solve the actual t=0 waveform. */
public class ReferenceTrajectoryPrepare20261018 {
 static double voltage(Model m){
  String tag="trajectoryVoltageAudit";m.result().numerical().create(tag,"EvalGlobal");
  m.result().numerical(tag).set("data","dset1");m.result().numerical(tag).set("solnum",1);
  // Evaluate the waveform separately: dset1 may retain a swept V_cell value.
  m.result().numerical(tag).set("expr",new String[]{"step_load(0[s])"});
  double wave=m.result().numerical(tag).getReal()[0][0];m.result().numerical().remove(tag);
  return m.param().evaluate("V_cell")-m.param().evaluate("V_step_amp")*wave;
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);
  Model m=ModelUtil.load("TrajectoryInitial",a[0]);ReferenceSteadySolve20261017.freezeLocalMesh(m);
  String mesh=ReferenceSteadySolve20261017.meshHash(m),field=LFFrozenRecovery.hash(m.sol("sol1").getU());
  double beforeV=voltage(m);double[] old={m.param().evaluate("T_in"),m.param().evaluate("lambda_air"),m.param().evaluate("FU")};
  m.param().set("V_cell",a[3]+"[V]");m.param().set("T_in",a[4]+"[K]");m.param().set("lambda_air",a[5]);m.param().set("FU",a[6]);m.param().set("V_step_amp",a[7]+"[V]");
  if(a[8].equals("legacy_step")){
   if(!m.func("step_load").getType().equals("Step"))throw new RuntimeException("Expected native Step function");
   m.func("step_load").set("location",a[9]+"[s]");m.func("step_load").set("smooth",.1);m.func("step_load").set("smoothactive",true);
  }else{
   m.func().remove("step_load");m.func().create("step_load","Analytic");m.func("step_load").set("funcname","step_load");
   m.func("step_load").set("args",new String[]{"t"});m.func("step_load").set("argunit",new String[]{"s"});m.func("step_load").set("fununit","1");
   m.func("step_load").set("expr",a[8].replaceAll("\\bt\\b","(t/1[s])"));
  }
  double actualV=voltage(m),expectedV=Double.parseDouble(a[10]);
  if(Math.abs(actualV-expectedV)>1e-12)throw new RuntimeException("Actual t=0 waveform voltage differs from planned steady initial voltage");
  boolean reuse=Math.abs(beforeV-actualV)<1e-12;
  for(int i=0;i<3;i++)reuse &= Math.abs(old[i]-Double.parseDouble(a[i+4]))<1e-12;
  if(reuse){if(!field.equals(LFFrozenRecovery.hash(m.sol("sol1").getU())))throw new RuntimeException("Reused steady field changed");}
  else{
   m.sol("sol1").copySolution("trajectoryInitialWarm");
   m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","trajectoryInitialWarm");m.sol("sol1").feature("v1").set("solnum","1");
   m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{a[3]}});
   System.out.println("INITIAL_STEADY_SOLVE_START");System.out.flush();m.sol("sol1").runAll();
  }
  if(!mesh.equals(ReferenceSteadySolve20261017.meshHash(m)))throw new RuntimeException("Initial mesh changed");
  m.label("Trajectory steady initial field at actual t=0 waveform; transient qualification pending");m.save(a[1]);
  System.out.println("TRAJECTORY_INITIAL_COMPLETE|reused="+reuse+"|initial_V="+actualV+"|mesh="+mesh);System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
