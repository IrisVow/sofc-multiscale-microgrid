/*===========================================================================
 * Solve3D - solve the SOFC 3D unit-cell model and export T fields.
 * Uses createAutoSequence + explicit solver tuning (V_cell continuation 1.0->0.75).
 * Usage: Solve3D steady | transient
 *===========================================================================*/
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;

public class Solve3D {
    static Model model;
    static final String MODEL = ProjectPaths.resolve("outputs/verify3d/sofc3d_model.mph");
    static final String OUT  = ProjectPaths.resolve("outputs/verify3d/out/");
    static final String[] EXPRS = {"x","y","z","dom","T"};
    static final String[][] SCALE = {
        {"comp1_p","1e5"},{"comp1_phil","1"},{"comp1_phis","1"},{"comp1_T","1e3"},
        {"comp1_u","10"},{"comp1_w1","1"},{"comp1_w2","1"},{"comp1_w3","1"}
    };
    static void p(String s) { System.out.println(s); System.out.flush(); }
    static boolean has(SolverSequence s, String tag) {
        for (String t : s.feature().tags()) if (t.equals(tag)) return true;
        return false;
    }
    static boolean has(SolverFeature f, String tag) {
        try { for (String t : f.feature().tags()) if (t.equals(tag)) return true; } catch (Throwable t) {}
        return false;
    }

    public static void main(String[] args) throws Exception {
        ModelUtil.initStandalone(false);
        if (args.length == 0) { p("usage: Solve3D steady|transient"); System.exit(0); }
        String phase = args[0];
        new File(OUT).mkdirs();
        model = ModelUtil.load("Model", MODEL);
        model.study("std1").setGenPlots(false);
        model.study("std1").setGenIntermediatePlots(false);
        model.study("std2").setGenPlots(false);
        model.study("std2").setGenIntermediatePlots(false);
        try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
        long t0 = System.currentTimeMillis();

        if (phase.equals("steady")) {
            setupSol1();
            p("== run sol1 (V_cell 1.0->0.75 continuation) ==");
            model.sol("sol1").runAll();
            p("  runAll done in " + (System.currentTimeMillis()-t0)/1000 + "s; datasets=" + java.util.Arrays.toString(model.result().dataset().tags()));
            // dset1 solnums: 1..6 -> V=1.0,0.95,0.9,0.85,0.8,0.75
            exportSnap("dset1", 1, OUT + "s_V1p0.csv");
            exportSnap("dset1", 5, OUT + "s_S5_V0p8.csv");
            exportSnap("dset1", 6, OUT + "s_S6_V0p75.csv");
            p("  exports done");
            model.save(ProjectPaths.resolve("outputs/verify3d/sofc3d_solved_steady.mph"));
        } else if (phase.equals("transient")) {
            model.param().set("V_cell", "1.0[V]");
            model.param().set("V_step_amp", "0.2[V]");
            try { model.func("step_load").set("location", "100[s]"); } catch (Throwable t) { p("  step loc ERR " + t); }
            setupSol1();
            p("== steady init (V=1.0) ==");
            model.sol("sol1").runAll();
            p("  steady done in " + (System.currentTimeMillis()-t0)/1000 + "s");
            exportSnap("dset1", 1, OUT + "t_init_steady.csv");
            p("  init exported; setting up sol2");
            setupSol2();
            p("== run sol2 (transient T3/T7) ==");
            model.sol("sol2").runAll();
            p("  transient done in " + (System.currentTimeMillis()-t0)/1000 + "s");
            String[] tlist = {"0","10","20","30","40","50","60","80","99","100","101","110","125","150","180","220","280","360","480","640","840","1200"};
            for (int i = 0; i < tlist.length; i++) {
                exportSnap("dset2", i + 1, OUT + "t_t" + tlist[i] + "s.csv");
                p("  t=" + tlist[i] + "s exported");
            }
            model.save(ProjectPaths.resolve("outputs/verify3d/sofc3d_solved_transient.mph"));
        }
        p("DONE time=" + (System.currentTimeMillis()-t0)/1000 + "s");
        System.exit(0);
    }

    static void setupSol1() {
        p("== setup sol1 ==");
        try {
            model.sol().create("sol1");
            model.sol("sol1").study("std1");
            model.sol("sol1").createAutoSequence("std1");
            p("  auto sequence: " + java.util.Arrays.toString(model.sol("sol1").feature().tags()));
        } catch (Throwable t) { p("  sol1 auto ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            SolverFeature s1 = model.sol("sol1").feature("s1");
            p("  s1 children before cleanup: " + java.util.Arrays.toString(s1.feature().tags()));
            // remove default (segregated) solve steps; keep dDef/aDef only
            for (String tag : s1.feature().tags()) {
                if (tag.equals("dDef") || tag.equals("aDef")) continue;
                try { s1.feature().remove(tag); } catch (Throwable t) { p("  remove " + tag + " ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
            }
            s1.feature("dDef").set("linsolver", "pardiso");
            SolverFeature p1 = s1.create("p1", "Parametric");
            p1.set("pname", new String[]{"V_cell"});
            p1.set("plistarr", new String[]{"1.0","0.95","0.9","0.85","0.8","0.75"});
            p1.set("punit", new String[]{"V"});
            SolverFeature fc1 = s1.create("fc1", "FullyCoupled");
            fc1.set("linsolver", "dDef");
            fc1.set("initstep", "0.001");
            fc1.set("minstep", "1e-6");
            fc1.set("maxiter", "300");
            fc1.set("jtech", "onevery");
            try { s1.set("plotfreq", "never"); } catch (Throwable t) { p("  s1 plotfreq ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
            try { s1.set("probefreq", "never"); } catch (Throwable t) {}
            try { p1.set("plot", "off"); } catch (Throwable t) {}
            try { p1.set("probefreq", "never"); } catch (Throwable t) {}
            p("  s1 configured: " + java.util.Arrays.toString(s1.feature().tags()));
        } catch (Throwable t) { p("  s1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
    }

    static void setupSol2() {
        p("== setup sol2 ==");
        try {
            model.sol().create("sol2");
            model.sol("sol2").study("std2");
            model.sol("sol2").createAutoSequence("std2");
            p("  auto sequence: " + java.util.Arrays.toString(model.sol("sol2").feature().tags()));
        } catch (Throwable t) { p("  sol2 auto ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            model.sol("sol2").feature("v1").set("initmethod", "sol");
            model.sol("sol2").feature("v1").set("initsol", "sol1");
            model.sol("sol2").feature("v1").set("solnum", 1);
            p("  v1 init from sol1 ok");
        } catch (Throwable t) { p("  v1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            SolverFeature t1 = model.sol("sol2").feature("t1");
            p("  t1 children: " + java.util.Arrays.toString(t1.feature().tags()));
            if (has(t1, "dDef")) t1.feature("dDef").set("linsolver", "pardiso");
            if (has(t1, "fcDef")) { try { t1.feature("fcDef").active(false); } catch (Throwable t) {} }
            if (!has(t1, "fc1")) {
                SolverFeature fc1 = t1.create("fc1", "FullyCoupled");
                fc1.set("linsolver", "dDef");
                fc1.set("initstep", "1e-3");
                fc1.set("minstep", "1e-6");
                fc1.set("maxiter", "100");
                fc1.set("jtech", "onevery");
            }
            p("  t1 configured: " + java.util.Arrays.toString(t1.feature().tags()));
        } catch (Throwable t) { p("  t1 ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
    }

    static void exportSnap(String dataset, int solnum, String fname) {
        try {
            model.result().export("data1").set("data", dataset);
            model.result().export("data1").set("solnum", solnum);
            model.result().export("data1").set("expr", EXPRS);
            model.result().export("data1").set("filename", fname);
            model.result().export("data1").run();
        } catch (Throwable t) { p("  export ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
    }
}
