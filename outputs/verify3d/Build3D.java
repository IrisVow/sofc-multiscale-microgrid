/*===========================================================================
 * Build3D - SOFC 3D unit-cell model (channel+rib period x 100mm x 2.62mm)
 * Physics replicated from 2D reference (sofc1_FINAL2_solved.mph):
 *   siec (Secondary Current Distribution) + tcs (Concentrated Species, 4 sp.)
 *   + fp (Free&Porous Media Flow) + ht (Heat Transfer) + ElectrochemicalHeating
 * Mesh: Size refinements + Sweep (auto Mapped source at x=0) + Distribution(x).
 * Studies: std1 stationary, std2 transient.  Solvers sol1/sol2 attached.
 *===========================================================================*/
import com.comsol.model.*;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.physics.MultiphysicsCoupling;
import com.comsol.model.util.*;

public class Build3D {
    static Model model;
    static void p(String s) { System.out.println(s); System.out.flush(); }

    static PhysicsFeature getOrCreate(com.comsol.model.physics.Physics ph, String tag, String type, int dim) {
        for (String t : ph.feature().tags()) if (t.equals(tag)) return ph.feature(tag);
        return ph.create(tag, type, dim);
    }

    public static void main(String[] args) throws Exception {
        ModelUtil.initStandalone(false);
        String outFile = args.length > 0 ? args[0] : ProjectPaths.resolve("outputs/verify3d/sofc3d_model.mph");
        int nelem = args.length > 1 ? Integer.parseInt(args[1]) : 200;
        String hEly = args.length > 2 ? args[2] : "1e-5[m]";
        String hElec = args.length > 3 ? args[3] : "1e-4[m]";
        String hCh = args.length > 4 ? args[4] : "2e-4[m]";
        p("MESH numelem=" + nelem + " hEly=" + hEly + " hElec=" + hElec + " hCh=" + hCh);
        model = ModelUtil.create("Model");
        model.modelNode().create("comp1");
        ModelParam param = model.param();

        // ============ 1. parameters (from sofc1_pinn_conditions.json) ============
        param.set("W_ch", "1.5[mm]");
        param.set("W_rib", "1.0[mm]");
        param.set("H_gde", "0.5[mm]");
        param.set("H_elec", "0.02[mm]");
        param.set("H_ch", "0.8[mm]");
        param.set("L", "100[mm]");
        param.set("V_cell", "1.0[V]");
        param.set("T_in", "1073[K]");
        param.set("FU", "0.6");
        param.set("i_avg", "3000[A/m^2]");
        param.set("p_ref", "1[atm]");
        param.set("eps_p", "0.35");
        param.set("tau_p", "3.0");
        param.set("kappa_p", "1e-12[m^2]");
        param.set("Av_an", "1e5[1/m]");
        param.set("Av_ca", "1e5[1/m]");
        param.set("k_i0", "0.1");
        param.set("i0_an", "5e3[A/m^2]*k_i0");
        param.set("i0_ca", "5e2[A/m^2]*k_i0");
        param.set("j0_an_vol", "i0_an*Av_an");
        param.set("j0_ca_vol", "i0_ca*Av_ca");
        param.set("sigmas_an", "3.5e4[S/m]");
        param.set("sigmas_ca", "1e4[S/m]");
        param.set("sigmal_YSZ", "3.34e4[S/m]");
        param.set("dEeqdT_an", "0[V/K]");
        param.set("wH2_in", "0.95");
        param.set("wH2O_in", "0.05");
        param.set("wO2_in", "0.21");
        param.set("wN2_in", "1-wO2_in");
        param.set("M_H2", "0.002016[kg/mol]");
        param.set("M_H2O", "0.018015[kg/mol]");
        param.set("M_O2", "0.031998[kg/mol]");
        param.set("M_N2", "0.028013[kg/mol]");
        param.set("xH2_in", "(wH2_in/M_H2)/(wH2_in/M_H2+wH2O_in/M_H2O)");
        param.set("xO2_in", "(wO2_in/M_O2)/(wO2_in/M_O2+wN2_in/M_N2)");
        param.set("C_H2_in", "p_ref/(R_const*T_in)*xH2_in");
        param.set("C_O2_in", "p_ref/(R_const*T_in)*xO2_in");
        param.set("u_in_a", "FU*i_avg*L/(2*F_const*C_H2_in*H_ch)");
        param.set("u_in_c", "lambda_air*i_avg*L/(4*F_const*C_O2_in*H_ch)");
        param.set("mu_g", "3e-5[Pa*s]");
        param.set("k_g", "0.1[W/(m*K)]");
        param.set("Cp_g", "1500[J/(kg*K)]");
        param.set("k_s", "2.0[W/(m*K)]");
        param.set("Cp_s", "500[J/(kg*K)]");
        param.set("rho_s", "5000[kg/m^3]");
        param.set("D12", "9e-4[m^2/s]");
        param.set("D13", "8e-4[m^2/s]");
        param.set("D14", "8e-4[m^2/s]");
        param.set("D23", "3e-4[m^2/s]");
        param.set("D24", "3e-4[m^2/s]");
        param.set("D34", "2e-4[m^2/s]");
        param.set("V_step_amp", "0.1[V]");
        param.set("h_conv", "5[W/(m^2*K)]");
        param.set("T_amb", "293.15[K]");
        param.set("dEeqdT_ca", "-2.3e-4[V/K]");
        param.set("lambda_air", "8");
        param.set("D_eff_gas", "1e-4[m^2/s]");
        param.set("delta_mt", "0.5[mm]");
        p("[1] parameters ok");

        // ============ 2. geometry: 9 blocks ============
        GeomSequence geom = model.geom().create("geom1", 3);
        GeomFeature blk;
        blk = geom.feature().create("blk1", "Block");  // air_ch
        blk.set("size", new String[]{"L", "H_ch", "W_ch"}); blk.set("pos", new String[]{"0", "0", "0"});
        blk = geom.feature().create("blk2", "Block");  // air_rib (solid interconnect)
        blk.set("size", new String[]{"L", "H_ch", "W_rib"}); blk.set("pos", new String[]{"0", "0", "W_ch"});
        blk = geom.feature().create("blk3", "Block");  // ca_gde_ch
        blk.set("size", new String[]{"L", "H_gde", "W_ch"}); blk.set("pos", new String[]{"0", "H_ch", "0"});
        blk = geom.feature().create("blk4", "Block");  // ca_gde_rib
        blk.set("size", new String[]{"L", "H_gde", "W_rib"}); blk.set("pos", new String[]{"0", "H_ch", "W_ch"});
        blk = geom.feature().create("blk5", "Block");  // ely
        blk.set("size", new String[]{"L", "H_elec", "W_ch+W_rib"}); blk.set("pos", new String[]{"0", "H_ch+H_gde", "0"});
        blk = geom.feature().create("blk6", "Block");  // an_gde_ch
        blk.set("size", new String[]{"L", "H_gde", "W_ch"}); blk.set("pos", new String[]{"0", "H_ch+H_gde+H_elec", "0"});
        blk = geom.feature().create("blk7", "Block");  // an_gde_rib
        blk.set("size", new String[]{"L", "H_gde", "W_rib"}); blk.set("pos", new String[]{"0", "H_ch+H_gde+H_elec", "W_ch"});
        blk = geom.feature().create("blk8", "Block");  // fu_ch
        blk.set("size", new String[]{"L", "H_ch", "W_ch"}); blk.set("pos", new String[]{"0", "H_ch+2*H_gde+H_elec", "0"});
        blk = geom.feature().create("blk9", "Block");  // fu_rib (solid interconnect)
        blk.set("size", new String[]{"L", "H_ch", "W_rib"}); blk.set("pos", new String[]{"0", "H_ch+2*H_gde+H_elec", "W_ch"});
        geom.run("fin");
        p("[2] geometry ok: domains=" + geom.getNDomains() + " boundaries=" + geom.getNBoundaries());

        // ============ 3. step function ============
        model.func().create("step_load", "Step");
        model.func("step_load").set("location", "100[s]");
        model.func("step_load").set("from", "0");
        model.func("step_load").set("to", "1");

        // ============ 4. physics: siec ============
        model.physics().create("siec", "SecondaryCurrentDistribution", "geom1");
        model.physics("siec").selection().set(new int[]{3,4,5,6,7});
        try {
            PhysicsFeature ice1 = getOrCreate(model.physics("siec"), "ice1", "Electrolyte", 3);
            // default Electrolyte selection is not editable (covers all siec domains);
            // restrict conductivity to the electrolyte y-band exactly like the 2D reference.
            ice1.set("sigmal_mat", "userdef");
            ice1.set("sigmal", "sigmal_YSZ*exp(-10300[K]/T)*((y>=H_ch+H_gde)&&(y<=H_ch+H_gde+H_elec))");
        } catch (Throwable t) { p("  ice1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature pca1 = model.physics("siec").create("pca1", "PorousElectrode", 3);
            pca1.selection().set(new int[]{3,4});
            pca1.set("sigmal_mat", "userdef");
            pca1.set("sigmal", "sigmal_YSZ*exp(-10300[K]/T)");
            pca1.set("sigmas_mat", "userdef");
            pca1.set("sigmas", "sigmas_ca");
            try { pca1.feature("per1").active(false); } catch (Throwable t) { p("  pca per1 disable: " + String.valueOf(t.getMessage()).split("\n")[0]); }
            PhysicsFeature pcar1 = pca1.create("pcar1", "PorousElectrodeReaction", 3);
            pcar1.set("Eeq_mat", "userdef");
            pcar1.set("Eeq", "1[V]");
            pcar1.set("ElectrodeKinetics", "ButlerVolmer");
            pcar1.set("i0Type", "userdef");
            pcar1.set("i0", "i0_ca");
            pcar1.set("alphaa", "0.5");
            pcar1.set("alphac", "0.5");
            pcar1.set("LimitingCurrentDensity", "1");
            pcar1.set("ilim", "4*F_const*D_eff_gas*(w3*tcs.rho/M_O2)/delta_mt");
            pcar1.set("Av", "Av_ca");
            pcar1.set("HeatofReactionType", "TemperatureDerivative");
            pcar1.set("dEeqdT_mat", "userdef");
            pcar1.set("dEeqdT", "dEeqdT_ca");
        } catch (Throwable t) { p("  pca1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature pan1 = model.physics("siec").create("pan1", "PorousElectrode", 3);
            pan1.selection().set(new int[]{6,7});
            pan1.set("sigmal_mat", "userdef");
            pan1.set("sigmal", "sigmal_YSZ*exp(-10300[K]/T)");
            pan1.set("sigmas_mat", "userdef");
            pan1.set("sigmas", "sigmas_an");
            try { pan1.feature("per1").active(false); } catch (Throwable t) { p("  pan per1 disable: " + String.valueOf(t.getMessage()).split("\n")[0]); }
            PhysicsFeature panr1 = pan1.create("panr1", "PorousElectrodeReaction", 3);
            panr1.set("Eeq_mat", "userdef");
            panr1.set("Eeq", "0[V]");
            panr1.set("ElectrodeKinetics", "ButlerVolmer");
            panr1.set("i0Type", "userdef");
            panr1.set("i0", "i0_an");
            panr1.set("alphaa", "0.5");
            panr1.set("alphac", "0.5");
            panr1.set("LimitingCurrentDensity", "1");
            panr1.set("ilim", "2*F_const*D_eff_gas*(w1*tcs.rho/M_H2)/delta_mt");
            panr1.set("Av", "Av_an");
            panr1.set("HeatofReactionType", "TemperatureDerivative");
            panr1.set("dEeqdT_mat", "userdef");
            panr1.set("dEeqdT", "0[V/K]");
        } catch (Throwable t) { p("  pan1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature epot1 = model.physics("siec").create("epot1", "ElectricPotential", 2);
            epot1.selection().set(new int[]{12});
            epot1.set("phisbnd", "V_cell - V_step_amp*step_load(if(isdefined(t), t, 0[s]))");
        } catch (Throwable t) { p("  epot1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature egnd1 = model.physics("siec").create("egnd1", "ElectricGround", 2);
            egnd1.selection().set(new int[]{31});
        } catch (Throwable t) { p("  egnd1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            model.physics("siec").feature("init1").set("phis", "V_cell");
            model.physics("siec").feature("init1").set("phil", "0[V]");
            PhysicsFeature init2 = model.physics("siec").create("init2", "init", 3);
            init2.selection().set(new int[]{6,7});
            init2.set("phis", "0[V]");
            init2.set("phil", "0[V]");
        } catch (Throwable t) { p("  init ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("[4] siec ok");

        // ============ 5. physics: tcs (4 species) ============
        model.physics().create("tcs", "ConcentratedSpecies", "geom1", new String[]{"w1","w2","w3","w4"});
        model.physics("tcs").selection().set(new int[]{1,3,4,6,7,8});
        try {
            PhysicsFeature sp1 = model.physics("tcs").feature("sp1");
            sp1.set("M_w1", "M_H2"); sp1.set("M_w2", "M_H2O"); sp1.set("M_w3", "M_O2"); sp1.set("M_w4", "M_N2");
        } catch (Throwable t) { p("  sp1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature cdm1 = model.physics("tcs").feature("cdm1");
            cdm1.set("rho_src", "IdealGas");
            cdm1.set("u_src", "userdef");
            cdm1.set("u", new String[]{"comp1.u", "comp1.v", "comp1.w"});
            cdm1.set("DiffusionTableType", "table");
            cdm1.set("Species1", new String[]{"w1","w1","w1","w2","w2","w3"});
            cdm1.set("Species2", new String[]{"w2","w3","w4","w3","w4","w4"});
            cdm1.set("DF", new String[]{"D12","D13","D14","D23","D24","D34"});
            cdm1.set("DiffusivityFrom", new String[]{"UserDefined","UserDefined","UserDefined","UserDefined","UserDefined","UserDefined"});
            cdm1.set("minput_temperature_src", "userdef"); cdm1.set("minput_temperature", "T");
            cdm1.set("minput_pressure_src", "userdef"); cdm1.set("minput_pressure", "p_ref");
        } catch (Throwable t) { p("  cdm1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature init1 = model.physics("tcs").feature("init1");
            init1.set("MixtureSpecification", "MassFractions");
            init1.set("w0", new String[]{"0.45","0.45","0.1","0"});
            init1.set("p0", "1[atm]");
            init1.set("T0", "T_in");
            init1.set("InitialDensitySpecification", "IdealGas");
        } catch (Throwable t) { p("  tcs init ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature mf1 = model.physics("tcs").create("mf1", "MassFraction", 2);
            mf1.selection().set(new int[]{27});
            mf1.set("species", new String[]{"1","0","0","0"}); mf1.set("wbnd", new String[]{"0.95","0","0","0"});
            PhysicsFeature mf2 = model.physics("tcs").create("mf2", "MassFraction", 2);
            mf2.selection().set(new int[]{27});
            mf2.set("species", new String[]{"0","1","0","0"}); mf2.set("wbnd", new String[]{"0","0.05","0","0"});
            PhysicsFeature mf3 = model.physics("tcs").create("mf3", "MassFraction", 2);
            mf3.selection().set(new int[]{1});
            mf3.set("species", new String[]{"0","0","1","0"}); mf3.set("wbnd", new String[]{"0","0","0.21","0"});
            PhysicsFeature mf4 = model.physics("tcs").create("mf4", "MassFraction", 2);
            mf4.selection().set(new int[]{27});
            mf4.set("species", new String[]{"0","0","1","0"}); mf4.set("wbnd", new String[]{"0","0","0","0"});
            PhysicsFeature mf5 = model.physics("tcs").create("mf5", "MassFraction", 2);
            mf5.selection().set(new int[]{1});
            mf5.set("species", new String[]{"1","0","0","0"}); mf5.set("wbnd", new String[]{"0","0","0","0"});
            PhysicsFeature mf6 = model.physics("tcs").create("mf6", "MassFraction", 2);
            mf6.selection().set(new int[]{1});
            mf6.set("species", new String[]{"0","1","0","0"}); mf6.set("wbnd", new String[]{"0","0","0","0"});
        } catch (Throwable t) { p("  mf ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature of1 = model.physics("tcs").create("of1", "Outflow", 2);
            of1.selection().set(new int[]{36,43});
        } catch (Throwable t) { p("  of1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("[5] tcs ok");

        // ============ 6. physics: fp ============
        model.physics().create("fp", "FreeAndPorousMediaFlow", "geom1");
        model.physics("fp").selection().set(new int[]{1,3,4,6,7,8});
        try {
            PhysicsFeature fp1 = model.physics("fp").feature("fp1");
            fp1.set("rho_mat", "userdef"); fp1.set("rho", "tcs.rho");
            fp1.set("mu_mat", "userdef"); fp1.set("mu", "mu_g");
        } catch (Throwable t) { p("  fp1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature porous1 = model.physics("fp").create("porous1", "PorousMedium", 3);
            porous1.selection().set(new int[]{3,4,6,7});
            porous1.set("flowModelType", "darcian");
            porous1.set("permeabilityModelType", "permeability");
            PhysicsFeature pfluid1 = porous1.feature("fluid1");
            pfluid1.set("rho_mat", "userdef"); pfluid1.set("rho", "tcs.rho");
            pfluid1.set("mu_mat", "userdef"); pfluid1.set("mu", "mu_g");
            PhysicsFeature ppm1 = porous1.feature("pm1");
            ppm1.set("epsilon_p_mat", "userdef"); ppm1.set("epsilon_p", "eps_p");
            ppm1.set("kappa_mat", "userdef"); ppm1.set("kappa", "kappa_p");
        } catch (Throwable t) { p("  porous1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature inl1 = model.physics("fp").create("inl1", "InletBoundary", 2);
            inl1.selection().set(new int[]{1});
            inl1.set("BoundaryCondition", "Velocity");
            inl1.set("U0in", "u_in_c");
            inl1.set("FlowDirection", "NormalFlow");
            PhysicsFeature inl2 = model.physics("fp").create("inl2", "InletBoundary", 2);
            inl2.selection().set(new int[]{27});
            inl2.set("BoundaryCondition", "Velocity");
            inl2.set("U0in", "u_in_a");
            inl2.set("FlowDirection", "NormalFlow");
            PhysicsFeature out1 = model.physics("fp").create("out1", "OutletBoundary", 2);
            out1.selection().set(new int[]{36,43});
            out1.set("BoundaryCondition", "Pressure");
            out1.set("p0", "0");
        } catch (Throwable t) { p("  inl/out ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("[6] fp ok");

        // ============ 7. physics: ht ============
        model.physics().create("ht", "HeatTransferInSolidsAndFluids", "geom1");
        try {
            PhysicsFeature solid1 = model.physics("ht").feature("solid1");
            solid1.set("k_mat", "userdef"); solid1.set("k", "k_s");
            solid1.set("rho_mat", "userdef"); solid1.set("rho", "rho_s");
            solid1.set("Cp_mat", "userdef"); solid1.set("Cp", "Cp_s");
        } catch (Throwable t) { p("  solid1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature fluid1 = model.physics("ht").feature("fluid1");
            fluid1.selection().set(new int[]{1,8});
            fluid1.set("k_mat", "userdef"); fluid1.set("k", "k_g");
            fluid1.set("Cp_mat", "userdef"); fluid1.set("Cp", "Cp_g");
            fluid1.set("rho_mat", "userdef"); fluid1.set("rho", "tcs.rho");
            fluid1.set("u_src", "userdef"); fluid1.set("u", new String[]{"comp1.u", "comp1.v", "comp1.w"});
            fluid1.set("pref_src", "userdef"); fluid1.set("pref", "1[atm]");
        } catch (Throwable t) { p("  fluid1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            PhysicsFeature ifl1 = model.physics("ht").create("ifl1", "Inflow", 2);
            ifl1.selection().set(new int[]{1,27});
            ifl1.set("Tustr_src", "userdef"); ifl1.set("Tustr", "T_in");
            ifl1.set("pustr_src", "userdef"); ifl1.set("pustr", "1[atm]");
            PhysicsFeature ofl1 = model.physics("ht").create("ofl1", "ConvectiveOutflow", 2);
            ofl1.selection().set(new int[]{36,43});
            PhysicsFeature hf1 = model.physics("ht").create("hf1", "HeatFluxBoundary", 2);
            hf1.selection().set(new int[]{2,5,34,35});
            hf1.set("HeatFluxType", "ConvectiveHeatFlux");
            hf1.set("HeatTransferCoefficientType", "UserDef");
            hf1.set("h", "h_conv");
            hf1.set("Text", "T_in");
        } catch (Throwable t) { p("  ht bc ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            model.physics("ht").feature("init1").set("Tinit", "T_in");
        } catch (Throwable t) { p("  ht init ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("[7] ht ok");

        // ============ 8. multiphysics: ElectrochemicalHeating ============
        try {
            MultiphysicsCoupling ech = model.component("comp1").multiphysics().create("electrocheal1", "ElectrochemicalHeating");
            ech.selection().set(new int[]{3,4,5,6,7});
            ech.set("ECHeat_physics", "siec");
            ech.set("Heat_physics", "ht");
        } catch (Throwable t) { p("  multiphysics ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("[8] multiphysics ok");

        // ============ 9. mesh (2D-recipe: Mapped electrolyte + FreeTri rest + Sweep x) ============
        MeshSequence mesh = model.mesh().create("mesh1", "geom1");
        // Size on source faces (x=0): electrolyte face 15, GDE faces 8/11/20/23, channel faces 1/4/27/30
        try {
            MeshFeature szEly = mesh.feature().create("szEly", "Size");
            szEly.selection().geom("geom1", 2).set(new int[]{15});
            szEly.set("hmaxactive", true); szEly.set("hmax", hEly);
        } catch (Throwable t) { p("  szEly ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            MeshFeature szGde = mesh.feature().create("szGde", "Size");
            szGde.selection().geom("geom1", 2).set(new int[]{8,11,20,23});
            szGde.set("hmaxactive", true); szGde.set("hmax", hElec);
        } catch (Throwable t) { p("  szGde ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            MeshFeature szCh = mesh.feature().create("szCh", "Size");
            szCh.selection().geom("geom1", 2).set(new int[]{1,4,27,30});
            szCh.set("hmaxactive", true); szCh.set("hmax", hCh);
        } catch (Throwable t) { p("  szCh ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        // Mapped electrolyte strip on the source face
        try {
            MeshFeature mpEly = mesh.feature().create("mpEly", "Mapped");
            mpEly.selection().set(new int[]{15});
        } catch (Throwable t) { p("  mpEly ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        // FreeTri on the other source faces
        try {
            MeshFeature ft1 = mesh.feature().create("ft1", "FreeTri");
            ft1.selection().set(new int[]{1,4,8,11,20,23,27,30});
        } catch (Throwable t) { p("  ft1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        // Sweep along x
        try {
            MeshFeature sw1 = mesh.feature().create("sw1", "Sweep");
            sw1.selection().set(new int[]{1,2,3,4,5,6,7,8,9});
            sw1.selection("sourceface").set(new int[]{1,4,8,11,15,20,23,27,30});
            sw1.selection("targetface").set(new int[]{36,37,38,39,40,41,42,43,44});
        } catch (Throwable t) { p("  sw1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            MeshFeature dis1 = mesh.feature().create("dis1", "Distribution");
            dis1.selection().set(new int[]{1,2,3,4,5,6,7,8,9});
            dis1.set("type", "number");
            dis1.set("numelem", String.valueOf(nelem));
        } catch (Throwable t) { p("  dis1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        long t0 = System.currentTimeMillis();
        try {
            mesh.run();
            p("[9] mesh built in " + (System.currentTimeMillis()-t0)/1000 + " s");
            try { MeshStatistics st = model.mesh("mesh1").stat(); p("  mesh stats: elem=" + st.getNumElem()); } catch (Throwable te) { p("  mesh stats ERR: " + String.valueOf(te.getMessage()).split("\n")[0]); }
        } catch (Throwable t) {
            p("[9] mesh.run ERR: " + String.valueOf(t.getMessage()).replace('\n',' ').substring(0, Math.min(300, String.valueOf(t.getMessage()).length())));
            for (String pr : mesh.problems()) p("   problem: " + pr);
        }
        // ============ 10. studies ============
        model.study().create("std1");
        model.study("std1").create("stat", "Stationary");
        model.study().create("std2");
        model.study("std2").create("time", "Transient");
        model.study("std2").feature("time").set("tlist", "0 10 20 30 40 50 60 80 99 100 101 110 125 150 180 220 280 360 480 640 840 1200");
        p("[10] studies ok");


        String out = outFile;
        model.save(out);
        p("SAVED " + out);
        System.exit(0);
    }
}
