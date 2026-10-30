import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class LFRecoveryPrototype{
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(a[2]);Model m=ModelUtil.load("LFRecovery",a[0]);
  m.component("comp1").physics().create("lfMass","GeneralFormPDE","geom1");m.physics("lfMass").selection().set(new int[]{1,2,4,5});
  m.physics("lfMass").field("dimensionless").field("lfR");m.physics("lfMass").field("dimensionless").component(new String[]{"lfR"});
  m.physics("lfMass").prop("Units").set("DependentVariableQuantity","none");m.physics("lfMass").prop("Units").set("CustomDependentVariableUnit","kg/m^3");m.physics("lfMass").prop("Units").set("SourceTermQuantity","none");m.physics("lfMass").prop("Units").set("CustomSourceTermUnit","kg/(m^3*s)");
  m.physics("lfMass").prop("ShapeProperty").set("order",1);m.physics("lfMass").prop("ShapeProperty").set("boundaryFlux",true);m.physics("lfMass").prop("ShapeProperty").set("boundaryFluxSmooth",false);
  m.physics("lfMass").feature("gfeq1").set("Ga",new String[]{"withsol('sol1',fp.rho*u)","withsol('sol1',fp.rho*v)"});m.physics("lfMass").feature("gfeq1").set("f","withsol('sol1',fp.Qm)-lfR/1[s]");m.physics("lfMass").feature("gfeq1").set("da","0");m.physics("lfMass").feature("gfeq1").set("ea","0");
  m.physics("lfMass").create("lfPorts","FluxBoundary",1);m.physics("lfMass").feature("lfPorts").selection().set(new int[]{1,9,12,16});m.physics("lfMass").feature("lfPorts").set("g","-withsol('sol1',fp.rho*(u*nx+v*ny))");m.physics("lfMass").feature("lfPorts").set("q","0");
  m.study().create("lfStudy");m.study("lfStudy").create("stat","Stationary");String[] tags=m.physics().tags();ArrayList<String> active=new ArrayList<>();for(String tag:tags){active.add(tag);active.add(tag.equals("lfMass")?"on":"off");}m.study("lfStudy").feature("stat").set("activate",active.toArray(new String[0]));
  m.study("lfStudy").createAutoSequences("all");
  for(String tag:m.sol().tags())if(m.sol(tag).study().equals("lfStudy")){System.out.println("AUX_SOLVER|"+tag);for(String f:m.sol(tag).feature().tags())System.out.println("AUX_FEATURE|"+f+"|"+m.sol(tag).feature(f).getType());}
  PrintWriter p=new PrintWriter(a[1]+"_equations.txt","UTF-8");for(String ft:m.physics("lfMass").feature().tags()){p.println("FEATURE|"+ft);for(String type:new String[]{"Weak","Expression","Shape"})for(String[] row:m.physics("lfMass").feature(ft).featureInfo("info").getInfoTable(type))p.println(Arrays.toString(row));}p.close();
  m.label("Frozen-field weak-equilibrium flux recovery prototype; not solved");m.save(a[1]+"_configured.mph");System.out.println("LF_CONFIGURED_NO_SOLVE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
