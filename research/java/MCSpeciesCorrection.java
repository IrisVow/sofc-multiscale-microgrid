import com.comsol.model.*;
import com.comsol.model.physics.*;
import java.util.*;

public class MCSpeciesCorrection {
 static final String RC="(d(fp.rho*u,x)+d(fp.rho*v,y)-fp.Qm)";
 static final int[] GAS={1,2,4,5}, WALL={3,6,7,8,13,15};
 static String volume(){StringJoiner s=new StringJoiner("+");for(String w:new String[]{"w1","w2","w3"})s.add("(-mc_alpha*"+w+"*"+RC+"*test("+w+")*tcs.d)");return s.toString();}
 static String boundary(){StringJoiner s=new StringJoiner("+");for(String w:new String[]{"w1","w2","w3"})s.add("(mc_alpha*"+w+"*fp.contCoeffFace*fp.unJump*test("+w+")*tcs.d)");return s.toString();}
 static String original(PhysicsFeature f,String name){for(String[] row:f.featureInfo("info").getInfoTable("Expression"))if(row.length>2&&name.equals(row[1]))return row[2];throw new IllegalStateException("Variable not found: "+name);}
 public static void apply(Model m,double alpha){
  m.param().set("mc_alpha",Double.toString(alpha),"Numerical species correction:0 original;1 full mass-consistent candidate");
  m.component("comp1").common().create("mcVolume","WeakContribution");m.component("comp1").common("mcVolume").selection().geom("geom1",2);m.component("comp1").common("mcVolume").selection().set(GAS);m.component("comp1").common("mcVolume").set("weakExpression",volume());m.component("comp1").common("mcVolume").set("integrationOrder","8");m.component("comp1").common("mcVolume").label("Species: volume continuity consistency correction");
  m.component("comp1").common().create("mcWall","WeakContribution");m.component("comp1").common("mcWall").selection().geom("geom1",1);m.component("comp1").common("mcWall").selection().set(WALL);m.component("comp1").common("mcWall").set("weakExpression",boundary());m.component("comp1").common("mcWall").set("integrationOrder","8");m.component("comp1").common("mcWall").label("Species: weak-wall numerical mass flux consistency");
  for(PhysicsFeature f:new PhysicsFeature[]{m.physics("tcs").feature("cdm1"),m.physics("tcs").feature("pom1").feature("fluid1")})for(String w:new String[]{"w1","w2","w3"}){
   String name="tcs.Res_"+w,orig=original(f,name),correct="("+orig+")+mc_alpha*"+w+"*"+RC;
   f.featureInfo("info").set(name,new String[]{correct});
   System.out.println("STABILIZATION_RESIDUAL_OVERRIDE|"+f.tag()+"|"+name+"|"+correct);
  }
 }
}
