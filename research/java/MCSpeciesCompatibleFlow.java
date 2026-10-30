import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
public class MCSpeciesCompatibleFlow {
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[3]);Model m=ModelUtil.load("MCCompatible",a[0]);double alpha=Double.parseDouble(a[2]);
  if(Math.abs(m.param().evaluate("alpha_rxn")-1)>1e-12)throw new IllegalStateException("Full source required");
  m.sol("sol1").copySolution("mcSameInitial");MCSpeciesCorrection.apply(m,alpha);
  m.physics("fp").prop("ShapeProperty").set("order_fluid",2);
  m.physics("fp").prop("ConsistentStabilization").set("StreamlineDiffusion",false);
  m.physics("fp").prop("ConsistentStabilization").set("CrosswindDiffusion",false);
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","mcSameInitial");m.sol("sol1").feature("v1").set("solnum","1");
  m.sol("sol1").feature("s1").set("stol","1e-5");m.sol("sol1").feature("s1").feature("fc1").set("initstep","0.1");m.sol("sol1").feature("s1").feature("fc1").set("minstep","1e-6");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","100");
  m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{Double.toString(m.param().evaluate("V_cell"))}});
  System.out.println("COMPATIBLE_FLOW_START|alpha="+alpha+"|vertices="+m.mesh("mesh1").getVertex()[0].length+"|order_fluid="+m.physics("fp").prop("ShapeProperty").getString("order_fluid"));System.out.flush();
  m.sol("sol1").runAll();m.label("P2/P1 Galerkin flow and mass-consistent species candidate; validation required");m.save(a[1]);m.save(a[1].replace(".mph","_history.java"),"java");System.out.println("COMPATIBLE_FLOW_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
