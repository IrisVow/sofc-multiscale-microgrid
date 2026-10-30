import com.comsol.model.*;import com.comsol.model.util.*;
public class ClosureTimeTolerance{
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);Model m=ModelUtil.load("TimeTolerance",a[0]);
  double old=m.sol("sol2").feature("t1").getDouble("rtol");
  System.out.println("RTOL_OLD|"+old);m.sol("sol2").feature("t1").set("rtol",old/10.);System.out.println("RTOL_NEW|"+(old/10.));System.out.flush();
  m.sol("sol2").runAll();System.out.println("TIME_TOLERANCE_SOLVE_COMPLETE");System.out.flush();m.save(a[1]);ModelUtil.remove("TimeTolerance");System.exit(0);
 }
}
