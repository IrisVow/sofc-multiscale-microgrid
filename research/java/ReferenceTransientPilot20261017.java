import com.comsol.model.*;import com.comsol.model.util.*;import com.comsol.model.physics.*;import java.io.*;import java.util.*;
/** Isolated constant-state transient qualification, never production reference data. */
public class ReferenceTransientPilot20261017{
 static boolean has(String[] a,String x){return Arrays.asList(a).contains(x);}
 static void audit(PhysicsFeature f,String path,PrintWriter p){p.println("FEATURE|"+path);try{for(String[] r:f.featureInfo("info").getInfoTable("Weak"))p.println(Arrays.toString(r));}catch(Throwable e){p.println(e);}for(String t:f.feature().tags())audit(f.feature(t),path+"/"+t,p);}
 static void assertReplacement(Model m){
  for(String ft:new String[]{"fp1","porous1"}){
   boolean convection=false;for(String[] r:m.physics("fp").feature(ft).featureInfo("info").getInfoTable("Weak")){if(r[1].contains("continuityEquation"))throw new RuntimeException("Native continuity not replaced in "+ft);if(r[1].contains("d(u")&&r[1].contains("test(u)"))convection=true;}
   if(!convection)throw new RuntimeException("Momentum convection inadvertently suppressed in "+ft);
  }
  for(PhysicsFeature f:new PhysicsFeature[]{m.physics("tbin").feature("cdm1"),m.physics("tbin").feature("pom1").feature("fluid1")})for(String[] r:f.featureInfo("info").getInfoTable("Weak"))if(!r[1].equals("0")&&!r[1].contains("cbf_"))throw new RuntimeException("Unreplaced native species term: "+r[1]);
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);
  Model m=ModelUtil.load("ReferenceTransientPilot",a[0]);
  int n=m.mesh("mesh1").getVertex()[0].length;String frozenMeshHash=ReferenceSteadySolve20261017.meshHash(m);System.out.println("PILOT_VERTICES|"+n);
  // Conservative bulk-volume storage. Derivatives vanish in the stationary study.
  m.component("comp1").common("lfCont").set("weakExpression","fp.rho*u*test(px)+fp.rho*v*test(py)+(fp.Qm-d(fp.epsilon_p*fp.rho,t))*test(p)");
  m.component("comp1").common("binarySpecies").set("weakExpression","((tcs2.j_bAx+fp.rho*u*bA)*test(bAx)+(tcs2.j_bAy+fp.rho*v*bA)*test(bAy)+(tcs2.R_bA-d(fp.epsilon_p*fp.rho*bA,t))*test(bA))*tcs2.d");
  ArrayList<String> active=new ArrayList<>();for(String t:m.physics().tags()){active.add(t);active.add(t.equals("tcs")?"off":"on");}
  m.study("std2").feature("time").set("activate",active.toArray(new String[0]));
  // Hold applied voltage constant at the existing stationary value for the invariance test.
  m.param().set("V_step_amp","0[V]");
  String times="0 0.01 0.1 1";if(a.length>4)times=a[4];
  m.study("std2").feature("time").set("tlist",times);
  m.sol().create("refTime");m.sol("refTime").study("std2");m.sol("refTime").attach("std2");
  m.sol("refTime").create("st1","StudyStep");m.sol("refTime").feature("st1").set("study","std2");m.sol("refTime").feature("st1").set("studystep","time");
  m.sol("refTime").create("v1","Variables");m.sol("refTime").feature("v1").set("control","time");
  m.sol("refTime").feature("v1").set("initmethod","sol");m.sol("refTime").feature("v1").set("initsol","sol1");m.sol("refTime").feature("v1").set("solnum","1");
  m.sol("refTime").feature("v1").set("notsolmethod","sol");m.sol("refTime").feature("v1").set("notsol","sol1");
  m.sol("refTime").create("t1","Time");SolverFeature t=m.sol("refTime").feature("t1");
  t.set("tlist",times);t.set("rtol","1e-4");t.set("maxorder",2);t.set("maxstepconstraintbdf","const");t.set("maxstepbdf","0.1");t.set("initialstepbdfactive",true);t.set("initialstepbdf","0.01");
  if(a.length>3&&a[3].contains("noinit"))t.set("consistent","off");
  t.create("d1","Direct");t.feature("d1").set("linsolver","pardiso");t.feature("d1").set("ooc","on");t.feature("d1").set("incore","manual");t.feature("d1").set("oocmemory",4096);
  if(a.length>3&&a[3].contains("mumps")){t.feature("d1").set("linsolver","mumps");t.feature("d1").set("oocmemory",3072);t.feature("d1").set("mumpsblr","off");}
  t.create("fc1","FullyCoupled");t.feature("fc1").set("linsolver","d1");if(has(t.feature().tags(),"fcDef"))t.feature().remove("fcDef");
  if(a.length>3&&a[3].startsWith("segregated")){
   t.feature().remove("fc1");t.create("seg1","Segregated");t.feature("seg1").set("maxsegiter",40);
   String[][] groups=a[3].contains("mass")?new String[][]{{"comp1_u","comp1_p","comp1_bA"},{"comp1_phis","comp1_phil"},{"comp1_T"}}:new String[][]{{"comp1_u","comp1_p"},{"comp1_bA","comp1_phis","comp1_phil"},{"comp1_T"}};
   for(int j=0;j<groups.length;j++){String tag="ss"+(j+1);t.feature("seg1").create(tag,"SegregatedStep");t.feature("seg1").feature(tag).set("segvar",groups[j]);t.feature("seg1").feature(tag).set("linsolver","d1");}
   if(has(t.feature("seg1").feature().tags(),"ssDef"))t.feature("seg1").feature().remove("ssDef");
  }
  m.sol("refTime").feature("v1").updateVariables();
  // Weak-row indices change between stationary and transient equation compilation.
  // Remove stationary locks first so a continuity lock cannot suppress momentum advection.
  for(String ft:new String[]{"fp1","porous1"}){
   PhysicsFeature f=m.physics("fp").feature(ft);f.featureInfo("info").clearLocks();
   for(String[] r:f.featureInfo("info").getInfoTable("Weak"))if(r[1].contains("continuityEquation"))f.featureInfo("info").set(r[2],new String[]{"0"});
  }
  for(PhysicsFeature f:new PhysicsFeature[]{m.physics("tbin").feature("cdm1"),m.physics("tbin").feature("pom1").feature("fluid1")}){
   f.featureInfo("info").clearLocks();for(String[] r:f.featureInfo("info").getInfoTable("Weak"))if(!r[1].contains("cbf_"))f.featureInfo("info").set(r[2],new String[]{"0"});
  }
  for(String w:new String[]{"w1","w2","w3"})if(has(m.sol("refTime").feature("v1").feature().tags(),"comp1_"+w))m.sol("refTime").feature("v1").feature("comp1_"+w).set("solvefor","off");
  if(!has(m.sol("refTime").feature("v1").feature().tags(),"comp1_bA"))throw new RuntimeException("Active binary field bA missing from transient solver");
  m.sol("refTime").feature("v1").feature("comp1_bA").set("solvefor","on");
  m.sol("refTime").feature("v1").feature("comp1_bA").set("scalemethod","manual");m.sol("refTime").feature("v1").feature("comp1_bA").set("scaleval","0.2");
  m.result().dataset().create("refTimeData","Solution");m.result().dataset("refTimeData").set("solution","refTime");
  assertReplacement(m);
  m.label("Transient qualification pilot; production status NOT ACCEPTED");m.save(a[1].replace(".mph","_configured.mph"));
  m.save(a[1].replace(".mph","_configured.java"),"java");
  PrintWriter p=new PrintWriter(a[1].replace(".mph","_weak_audit.txt"),"UTF-8");
  for(String ph:new String[]{"fp","tbin","ht"})for(String ft:m.physics(ph).feature().tags()){
   audit(m.physics(ph).feature(ft),ph+"/"+ft,p);
  }p.close();
  if(a.length>3&&a[3].equals("build")){System.out.println("PILOT_BUILD_COMPLETE");System.exit(0);}
  System.out.println("PILOT_SOLVE_START");System.out.flush();m.sol("refTime").runAll();assertReplacement(m);if(!frozenMeshHash.equals(ReferenceSteadySolve20261017.meshHash(m)))throw new RuntimeException("Frozen mesh changed during transient solve");m.save(a[1]);System.out.println("PILOT_SOLVE_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
