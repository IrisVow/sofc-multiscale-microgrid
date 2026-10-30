import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;import java.security.*;import java.nio.*;
public class ReferenceSteadySolve20261017{
 static String meshHash(Model m)throws Exception{MessageDigest h=MessageDigest.getInstance("SHA-256");ByteBuffer b=ByteBuffer.allocate(8);for(double[] col:m.mesh("mesh1").getVertex())for(double x:col){b.clear();b.putDouble(x);h.update(b.array());}String[] types=m.mesh("mesh1").getTypes();Arrays.sort(types);for(String type:types){h.update(type.getBytes("UTF-8"));for(int[] row:m.mesh("mesh1").getElem(type))for(int x:row){b.clear();b.putLong(x);h.update(b.array());}for(int x:m.mesh("mesh1").getElemEntity(type)){b.clear();b.putLong(x);h.update(b.array());}}StringBuilder s=new StringBuilder();for(byte x:h.digest())s.append(String.format("%02x",x));return s.toString();}
 static void freezeLocalMesh(Model m)throws Exception{
  if(!Arrays.asList(m.mesh("mesh1").feature().tags()).contains("localAdapt17"))return;
  if(!Arrays.asList(m.sol().tags()).contains("localMeshWarm17"))throw new RuntimeException("Saved pre-adaptation M2 solution is unavailable");
  String before=meshHash(m);m.mesh("mesh1").feature("localAdapt17").set("solution","localMeshWarm17");m.mesh("mesh1").run();
  if(!meshHash(m).equals(before))throw new RuntimeException("Rebinding adaptation changed frozen mesh nodes/connectivity/domain IDs");
  System.out.println("FROZEN_MESH_REBUILT_IDENTICALLY|source=localMeshWarm17|sha256="+before);System.out.flush();
 }
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("ReferenceSteady",a[0]);
  if(m.mesh("mesh1").getVertex()[0].length!=430239)throw new RuntimeException("Incorrect frozen mesh");freezeLocalMesh(m);String before=meshHash(m);
  m.sol("sol1").copySolution("referenceCaseInitial");
  m.param().set("V_cell",a[3]+"[V]");m.param().set("T_in",a[4]+"[K]");m.param().set("lambda_air",a[5]);m.param().set("FU",a[6]);m.param().set("V_step_amp","0[V]");
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","referenceCaseInitial");m.sol("sol1").feature("v1").set("solnum","1");
  m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{a[3]}});
  m.label("Configured corrected reference case; not yet solved or accepted");m.save(a[1].replace(".mph","_configured.mph"));
  System.out.println("REFERENCE_STEADY_START|"+Arrays.toString(a));System.out.flush();m.sol("sol1").runAll();
  if(!meshHash(m).equals(before))throw new RuntimeException("Frozen mesh changed during case solve");
  m.label("Corrected reference candidate; physical acceptance pending");m.save(a[1]);
  System.out.println("REFERENCE_STEADY_COMPLETE|mesh_sha256="+before);System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
