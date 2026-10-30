import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;import java.io.*;
public class MCSpeciesPrototype{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("MCPrototype",a[0]);MCSpeciesCorrection.apply(m,1.);
  for(String tag:new String[]{"mcVolume","mcWall"}){System.out.println("COMMON|"+tag+"|"+Arrays.toString(m.component("comp1").common(tag).selection().entities()));for(String key:m.component("comp1").common(tag).properties())try{System.out.println("PROPERTY|"+tag+"|"+key+"|"+Arrays.toString(m.component("comp1").common(tag).getStringArray(key)));}catch(Throwable e){}}
  for(com.comsol.model.physics.PhysicsFeature f:new com.comsol.model.physics.PhysicsFeature[]{m.physics("tcs").feature("cdm1"),m.physics("tcs").feature("pom1").feature("fluid1")})for(String w:new String[]{"w1","w2","w3"})System.out.println("LOCKED_RESIDUAL|"+f.tag()+"|"+MCSpeciesCorrection.original(f,"tcs.Res_"+w));
  m.label("Configured mass-consistent candidate; not recomputed");m.save(a[1]);m.save(a[2],"java");System.out.println("PROTOTYPE_READY_NO_SOLVE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
