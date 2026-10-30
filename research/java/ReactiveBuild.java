import com.comsol.model.*;import com.comsol.model.util.*;import java.util.*;
public class ReactiveBuild{
 public static void main(String[] a){try{run(a);}catch(Throwable e){System.out.println("BUILD_FAILED");e.printStackTrace(System.out);System.out.flush();System.exit(1);}}
 static void run(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showProgress(false);ModelUtil.showPlots(false);System.out.println("INIT_READY");System.out.flush();Model m=ModelUtil.load("ReactiveBuild",a[0]);System.out.println("MODEL_LOADED");System.out.flush();
  m.label("SOFC reactive transport candidate 2026-10-16");
  m.param().set("alpha_rxn","0","Solver continuation of species and net gas mass sources; final value must be1");
  m.param().set("u_in_a","i_avg*L/(2*F_const*C_H2_in*H_ch*FU)","Nominal design fuel feed; actual utilization is reported from solved consumption");
  System.out.println("INLET_CONFIGURED");System.out.flush();
  m.physics("tcs").prop("TransportMechanism").set("MassTransferInPorousMedia",true);
  System.out.println("POROUS_ROOT_ENABLED|"+Arrays.toString(m.physics("tcs").feature().tags()));System.out.flush();
  m.physics("tcs").feature("pom1").active(true);m.physics("tcs").feature("pom1").feature("fluid1").active(true);m.physics("tcs").feature("pom1").feature("pm1").active(true);
  m.physics("tcs").feature("pom1").feature("pm1").set("epsilon_p_mat","userdef");m.physics("tcs").feature("pom1").feature("pm1").set("epsilon_p","eps_p");
  m.physics("tcs").feature("pom1").feature("fluid1").set("FluidDiffusivityModelType","TortuosityModel");m.physics("tcs").feature("pom1").feature("fluid1").set("tauF","tau_p");
  // cdm1 is the locked default material feature; pom1 overrides domains2/4.
  System.out.println("POROUS_CONFIGURED");System.out.flush();
  for(String side:new String[]{"an","ca"}){
   String tag="rxn_"+side;m.physics("tcs").create(tag,"ReactionSources",2);m.physics("tcs").feature(tag).label("Faradaic mass sources "+side+" (total electrode volume)");
   m.physics("tcs").feature(tag).selection().set(new int[]{side.equals("an")?4:2});
   m.physics("tcs").feature(tag).set("MassTransferToOtherPhases",true);m.physics("tcs").feature(tag).set("ReactingVolumeType","TotalVolume");
   for(String s:new String[]{"w1","w2","w3","w4"})m.physics("tcs").feature(tag).set("R_"+s,"0[kg/(m^3*s)]");
   if(side.equals("an")){
    m.physics("tcs").feature(tag).set("R_w1","-alpha_rxn*cd.iloc_panr1*Av_an*M_H2/(2*F_const)");
    m.physics("tcs").feature(tag).set("R_w2","alpha_rxn*cd.iloc_panr1*Av_an*M_H2O/(2*F_const)");
   }else m.physics("tcs").feature(tag).set("R_w3","alpha_rxn*cd.iloc_pcar1*Av_ca*M_O2/(4*F_const)");
  }
  System.out.println("SOURCES_CONFIGURED");System.out.flush();
  // Composition and temperature change density. Preserve the reference thermodynamic-pressure convention.
  m.physics("fp").prop("PhysicalModelProperty").set("Compressibility","WeaklyCompressible");
  m.component("comp1").multiphysics().create("rxnflow","ReactingFlow");m.component("comp1").multiphysics("rxnflow").label("Species-to-flow mass coupling; no added heat coupling");
  m.component("comp1").multiphysics("rxnflow").set("Fluid_physics","fp");m.component("comp1").multiphysics("rxnflow").set("Species_physics","tcs");
  System.out.println("FLOW_COUPLING_CONFIGURED");System.out.flush();
  m.component("comp1").variable().create("rxnaud_an");m.component("comp1").variable("rxnaud_an").selection().geom("geom1",2);m.component("comp1").variable("rxnaud_an").selection().set(new int[]{4});
  m.component("comp1").variable("rxnaud_an").set("rxn_qan","cd.iloc_panr1*Av_an");
  m.component("comp1").variable("rxnaud_an").set("rxn_mdot_an","alpha_rxn*rxn_qan*(M_H2O-M_H2)/(2*F_const)");
  m.component("comp1").variable().create("rxnaud_ca");m.component("comp1").variable("rxnaud_ca").selection().geom("geom1",2);m.component("comp1").variable("rxnaud_ca").selection().set(new int[]{2});
  m.component("comp1").variable("rxnaud_ca").set("rxn_qca","cd.iloc_pcar1*Av_ca");m.component("comp1").variable("rxnaud_ca").set("rxn_mdot_ca","alpha_rxn*rxn_qca*M_O2/(4*F_const)");
  if(!m.physics("tcs").feature("pom1").isActive())throw new RuntimeException("Porous species inactive");
  m.sol("sol1").copySolution("parentWarm");m.param().set("V_cell","0.8[V]");
  System.out.println("WARM_STATE_COPIED");System.out.flush();
  m.sol("sol1").feature("v1").set("initmethod","sol");m.sol("sol1").feature("v1").set("initsol","parentWarm");m.sol("sol1").feature("v1").set("solnum","5");
  m.sol("sol1").feature("s1").feature("p1").set("pname",new String[]{"V_cell"});m.sol("sol1").feature("s1").feature("p1").set("plist",new String[][]{{"0.8"}});
  m.sol("sol1").feature("s1").set("stol","1e-5");
  System.out.println("SAVE_PREPARED_START");System.out.flush();m.save(a[1]);System.out.println("SAVE_PREPARED_DONE");System.out.flush();m.save(a[2],"java");
  System.out.println("BRANCH_PREPARED|vertices="+m.mesh("mesh1").getVertex()[0].length+"|porous_species="+m.physics("tcs").feature("pom1").isActive());
  for(String ph:m.component("comp1").multiphysics().tags())System.out.println("ACTIVE_COUPLING|"+ph+"|"+m.component("comp1").multiphysics(ph).getType());
  ModelUtil.remove("ReactiveBuild");System.out.println("NO_SOLVE_IN_PREPARATION");System.exit(0);
 }
}
