import com.comsol.model.*;import com.comsol.model.util.*;
public class ClosureRefine{
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);Model m=ModelUtil.load("ClosureRefine",a[0]);
  System.out.println("REFINE_START|vertices="+m.mesh("mesh1").getVertex()[0].length);System.out.flush();
  m.mesh("mesh1").create("closureRefine","Refine");m.mesh("mesh1").feature("closureRefine").set("rmethod","regular");m.mesh("mesh1").feature("closureRefine").set("numrefine",1);m.mesh("mesh1").run();
  System.out.println("MESH_REFINED|vertices="+m.mesh("mesh1").getVertex()[0].length);System.out.flush();
  m.sol("sol1").runAll();System.out.println("REFINED_SOLVE_COMPLETE|original_six_voltage_solver_sol1");System.out.flush();m.save(a[1]);ModelUtil.remove("ClosureRefine");System.exit(0);
 }
}
