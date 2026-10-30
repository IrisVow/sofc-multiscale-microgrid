import com.comsol.model.*;import com.comsol.model.util.*;
public class ClosureRefineWarm{
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("ClosureRefineWarm",a[0]);
  m.sol("sol1").copySolution("closureWarm");
  System.out.println("BASE_VERTICES|"+m.mesh("mesh1").getVertex()[0].length);
  m.mesh("mesh1").create("closureRefine","Refine");m.mesh("mesh1").feature("closureRefine").set("rmethod","regular");m.mesh("mesh1").feature("closureRefine").set("numrefine",1);m.mesh("mesh1").run();
  System.out.println("FINE_VERTICES|"+m.mesh("mesh1").getVertex()[0].length);
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","closureWarm");m.sol("sol1").feature("v1").set("solnum","6");
  m.param().set("V_cell","0.75[V]");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{"0.75"}});
  System.out.println("SOLVE_FINE_S6_START");System.out.flush();m.sol("sol1").runAll();m.save(a[1]+"_S6.mph");System.out.println("FINE_S6_SAVED");System.out.flush();
  m.sol("sol1").copySolution("closureFineS6");m.sol("sol1").feature("v1").set("initsol","closureFineS6");m.sol("sol1").feature("v1").set("solnum","1");
  m.param().set("V_cell","0.8[V]");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{"0.8"}});
  System.out.println("SOLVE_FINE_S5_START");System.out.flush();m.sol("sol1").runAll();m.save(a[1]+"_S5.mph");System.out.println("FINE_S5_SAVED");System.out.flush();ModelUtil.remove("ClosureRefineWarm");System.exit(0);
 }
}
