import com.comsol.model.*;
import com.comsol.model.util.*;

public class FluxMeshRefine {
 static Model m;static String prefix;static int step=0;
 static void solve(double v)throws Exception{
  m.param().set("V_cell",v+"[V]");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{Double.toString(v)}});
  System.out.println("REFINED_SOLVE_START|V="+v);System.out.flush();m.sol("sol1").runAll();
  if(Math.abs(v-.8)<1e-10||Math.abs(v-.75)<1e-10){String cid=v>.79?"S5":"S6";m.label("Flux/mesh audit M1 "+cid+" candidate; validation pending");m.save(prefix+"_"+cid+".mph");System.out.println("REFINED_MODEL_SAVED|"+cid);System.out.flush();}
  String tag="fluxMeshWarm"+step++;m.sol("sol1").copySolution(tag);m.sol("sol1").feature("v1").set("initsol",tag);m.sol("sol1").feature("v1").set("solnum","1");
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);m=ModelUtil.load("FluxMeshAudit",a[0]);prefix=a[1];
  if(Math.abs(m.param().evaluate("alpha_rxn")-1)>1e-12)throw new RuntimeException("Full source required");
  m.sol("sol1").copySolution("fluxMeshCoarseWarm");System.out.println("BASE_VERTICES|"+m.mesh("mesh1").getVertex()[0].length);System.out.flush();
  m.mesh("mesh1").create("fluxAuditRefinement","Refine");m.mesh("mesh1").feature("fluxAuditRefinement").set("rmethod","regular");m.mesh("mesh1").feature("fluxAuditRefinement").set("numrefine",1);m.mesh("mesh1").run();
  System.out.println("REFINED_VERTICES|"+m.mesh("mesh1").getVertex()[0].length);System.out.flush();
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","fluxMeshCoarseWarm");m.sol("sol1").feature("v1").set("solnum","1");
  m.sol("sol1").feature("s1").set("stol","1e-5");m.sol("sol1").feature("s1").feature("fc1").set("initstep","0.1");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","100");
  for(int k=0;k<=5;k++)solve((80-k)/100.);
  ModelUtil.remove("FluxMeshAudit");System.out.println("MESH_REFINEMENT_SOLVES_COMPLETE");System.exit(0);
 }catch(Throwable e){System.out.println("MESH_REFINEMENT_FAILED");e.printStackTrace(System.out);System.out.flush();System.exit(1);}}
}
