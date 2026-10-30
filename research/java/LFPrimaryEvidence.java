import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class LFPrimaryEvidence extends ReactiveExport{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);m=ModelUtil.load("LFEvidence",a[0]);data="dset1";sol=1;cid=a[2];summary(a[1]);PrintWriter p=new PrintWriter(a[1]+"_weak_evidence.txt","UTF-8");
 for(com.comsol.model.physics.PhysicsFeature f:new com.comsol.model.physics.PhysicsFeature[]{m.physics("fp").feature("fp1"),m.physics("fp").feature("porous1"),m.physics("tcs").feature("cdm1"),m.physics("tcs").feature("pom1").feature("fluid1")}){p.println("PARENT|"+f.tag());for(String[] row:f.featureInfo("info").getInfoTable("Weak"))p.println(Arrays.toString(row));}
 for(String tag:new String[]{"mcVolume","mcWall","lfCont","lfContPorts","lfSpecies","lfSpeciesPorts"}){p.println("COMMON|"+tag+"|active="+m.component("comp1").common(tag).isActive());p.println(m.component("comp1").common(tag).getString("weakExpression"));p.println("QUADRATURE|"+m.component("comp1").common(tag).getString("integrationOrder"));}p.close();System.out.println("PRIMARY_EVIDENCE_EXPORTED");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
