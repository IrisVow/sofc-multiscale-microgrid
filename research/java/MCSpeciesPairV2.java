import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
public class MCSpeciesPairV2 {
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[3]);Model m=ModelUtil.load("MCPair",a[0]);
  double alpha=Double.parseDouble(a[2]);if(Math.abs(m.param().evaluate("alpha_rxn")-1)>1e-12)throw new IllegalStateException("Full source required");
  m.sol("sol1").copySolution("mcSameInitial");MCSpeciesCorrection.apply(m,alpha);
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","mcSameInitial");m.sol("sol1").feature("v1").set("solnum","1");
  m.sol("sol1").feature("s1").set("stol","1e-5");m.sol("sol1").feature("s1").feature("fc1").set("initstep","0.1");m.sol("sol1").feature("s1").feature("fc1").set("minstep","1e-6");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","100");
  double v=m.param().evaluate("V_cell");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{Double.toString(v)}});
  System.out.println("PAIR_START|alpha="+alpha+"|voltage="+v+"|vertices="+m.mesh("mesh1").getVertex()[0].length);System.out.flush();long start=System.nanoTime();
  m.sol("sol1").runAll();m.label("Mass-consistent species paired candidate alpha="+alpha+"; validation required");m.save(a[1]);m.save(a[1].replace(".mph","_history.java"),"java");
  System.out.println("PAIR_COMPLETE|seconds="+(System.nanoTime()-start)/1e9+"|output="+a[1]);System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
