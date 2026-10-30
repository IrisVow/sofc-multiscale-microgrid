/*===========================================================================
 * Solve3DV2 - solve SOFC 3D unit-cell verification cases (S5/S6 steady,
 *             T3/T7 transient) using study.run() auto solvers.
 * Usage: Solve3DV2 steady | transient
 *===========================================================================*/
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;

public class Solve3DV2 {
    static Model model;
    static final String BASE = ProjectPaths.resolve("outputs/verify3d/sofc3d_m2.mph");
    static final String OUT  = ProjectPaths.resolve("outputs/verify3d/out/");
    static final String[] EXPRS = {"x","y","z","dom","T"};
    static void p(String s) { System.out.println(s); System.out.flush(); }

    public static void main(String[] args) throws Exception {
        ModelUtil.initStandalone(false);
        if (args.length == 0) { p("usage: Solve3DV2 steady|transient"); System.exit(0); }
        String phase = args[0];
        new File(OUT).mkdirs();
        long t0 = System.currentTimeMillis();
        if (phase.equals("steady")) {
            solveSteadyV("1.0", OUT + "s_V1p0.csv");
            solveSteadyV("0.8", OUT + "s_S5_V0p8.csv");
            solveSteadyV("0.75", OUT + "s_S6_V0p75.csv");
        } else if (phase.equals("transient")) {
            runTransient("0.2", "100", "T3_T7");
        }
        p("DONE total=" + (System.currentTimeMillis()-t0)/1000 + "s");
        System.exit(0);
    }

    static void solveSteadyV(String V, String fname) throws Exception {
        model = ModelUtil.load("Model", BASE);
        model.study("std1").setGenPlots(false);
        model.study("std1").setGenIntermediatePlots(false);
        model.study("std2").setGenPlots(false);
        model.study("std2").setGenIntermediatePlots(false);
        model.param().set("V_cell", V + "[V]");
        p("== steady V_cell=" + V + " ==");
        long t1 = System.currentTimeMillis();
        model.study("std1").run();
        p("  std1 run done in " + (System.currentTimeMillis()-t1)/1000 + "s");
        try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
        model.result().export("data1").set("data", "dset1");
        model.result().export("data1").set("solnum", 1);
        model.result().export("data1").set("expr", EXPRS);
        model.result().export("data1").set("filename", fname);
        model.result().export("data1").run();
        p("  exported " + fname);
        // Tmax probe
        try {
            model.result().numerical().create("mxT", "MaxSurface");
            model.result().numerical("mxT").set("data", "dset1");
            model.result().numerical("mxT").set("expr", new String[]{"T"});
            double[][] vv = model.result().numerical("mxT").getReal();
            p("  Tmax(V=" + V + ")=" + vv[0][0]);
        } catch (Throwable t) { p("  Tmax ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
    }

    static void runTransient(String amp, String stepT, String tag) throws Exception {
        model = ModelUtil.load("Model", BASE);
        model.study("std1").setGenPlots(false);
        model.study("std1").setGenIntermediatePlots(false);
        model.study("std2").setGenPlots(false);
        model.study("std2").setGenIntermediatePlots(false);
        model.param().set("V_cell", "1.0[V]");
        model.param().set("V_step_amp", amp + "[V]");
        try { model.func("step_load").set("location", stepT + "[s]"); } catch (Throwable t) { p("  step loc ERR: " + t); }
        p("== transient init steady V=1.0 (stepT=" + stepT + "s, amp=" + amp + "V) ==");
        long t1 = System.currentTimeMillis();
        model.study("std1").run();
        p("  std1 run done in " + (System.currentTimeMillis()-t1)/1000 + "s");
        try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
        model.result().export("data1").set("data", "dset1");
        model.result().export("data1").set("solnum", 1);
        model.result().export("data1").set("expr", EXPRS);
        model.result().export("data1").set("filename", OUT + "t_init_steady.csv");
        model.result().export("data1").run();
        p("  init exported");
        // configure transient study init from sol1
        try {
            model.study("std2").feature("time").set("initmethod", "sol");
            model.study("std2").feature("time").set("initstudy", "std1");
            model.study("std2").feature("time").set("initsol", "sol1");
            model.study("std2").feature("time").set("initsoluse", "sol1");
        } catch (Throwable t) { p("  std2 init ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("== run std2 transient ==");
        t1 = System.currentTimeMillis();
        model.study("std2").run();
        p("  std2 run done in " + (System.currentTimeMillis()-t1)/1000 + "s");
        String[] tlist = {"0","10","20","30","40","50","60","80","99","100","101","110","125","150","180","220","280","360","480","640","840","1200"};
        for (int i = 0; i < tlist.length; i++) {
            exportSnap("dset2", i + 1, OUT + "t_t" + tlist[i] + "s.csv");
        }
        p("  all " + tlist.length + " snapshots exported");
    }

    static void exportSnap(String dataset, int solnum, String fname) {
        try {
            model.result().export("data1").set("data", dataset);
            model.result().export("data1").set("solnum", solnum);
            model.result().export("data1").set("expr", EXPRS);
            model.result().export("data1").set("filename", fname);
            model.result().export("data1").run();
            p("  exported " + fname);
        } catch (Throwable t) { p("  export ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
    }
}
