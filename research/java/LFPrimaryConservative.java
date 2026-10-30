import com.comsol.model.*;import com.comsol.model.util.*;import com.comsol.model.physics.*;import java.util.*;import java.io.*;
public class LFPrimaryConservative{
 static void weak(Model m,String tag,int dim,int[] sel,String expr){m.component("comp1").common().create(tag,"WeakContribution");m.component("comp1").common(tag).selection().geom("geom1",dim);m.component("comp1").common(tag).selection().set(sel);m.component("comp1").common(tag).set("weakExpression",expr);m.component("comp1").common(tag).set("integrationOrder","8");}
 static void zero(PhysicsFeature f,boolean continuity){for(String[] row:f.featureInfo("info").getInfoTable("Weak")){String e=row[1];boolean target=continuity?e.contains("continuityEquation"):e.contains("tcs.j_w")||e.contains("-tcs.rho*(tcs.u*")||e.contains("tcs.R_w");if(target){System.out.println("ZERO_ORIGINAL_WEAK|"+row[2]+"|"+e);f.featureInfo("info").set(row[2],new String[]{"0"});}}
 }
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("LFPrimary",a[0]);m.sol("sol1").copySolution("lfSameInitial");
  for(String tag:new String[]{"mcVolume","mcWall"})m.component("comp1").common(tag).active(false);
  zero(m.physics("fp").feature("fp1"),true);zero(m.physics("fp").feature("porous1"),true);zero(m.physics("tcs").feature("cdm1"),false);zero(m.physics("tcs").feature("pom1").feature("fluid1"),false);
  PhysicsFeature wall=m.physics("fp").feature("wallbc1");for(int k=0;k<2;k++)wall.featureInfo("info").set("fp.upwindCont",k,new String[]{"0"});
  int[] gas={1,2,4,5},ports={1,9,12,16};weak(m,"lfCont",2,gas,"fp.rho*u*test(px)+fp.rho*v*test(py)+fp.Qm*test(p)");weak(m,"lfContPorts",1,ports,"-fp.rho*(u*nx+v*ny)*test(p)");
  StringJoiner volume=new StringJoiner("+"),boundary=new StringJoiner("+");for(String w:new String[]{"w1","w2","w3"}){volume.add("((tcs.j_"+w+"x+fp.rho*u*"+w+")*test("+w+"x)+(tcs.j_"+w+"y+fp.rho*v*"+w+")*test("+w+"y)+tcs.R_"+w+"*test("+w+"))*tcs.d");boundary.add("(-fp.rho*"+w+"*(u*nx+v*ny)*test("+w+"))*tcs.d");}
  weak(m,"lfSpecies",2,gas,volume.toString());weak(m,"lfSpeciesPorts",1,ports,boundary.toString());
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","lfSameInitial");m.sol("sol1").feature("v1").set("solnum","1");m.sol("sol1").feature("s1").set("stol","1e-7");m.sol("sol1").feature("s1").feature("fc1").set("initstep","0.1");m.sol("sol1").feature("s1").feature("fc1").set("maxiter","100");m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{Double.toString(m.param().evaluate("V_cell"))}});
  System.out.println("PRIMARY_FULL_CONSERVATIVE_START");System.out.flush();m.sol("sol1").runAll();m.label("Unified conservative mass/species weak forms, quadrature8; numerical validation required");m.save(a[1]);m.save(a[1].replace(".mph","_history.java"),"java");System.out.println("PRIMARY_FULL_CONSERVATIVE_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
