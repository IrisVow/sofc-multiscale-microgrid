/* EmsRecheck3D - COMSOL 3D steady recheck of EMS derate-triggered operating
 * points (T_in=1123 K, lambda_air=6, matching 2D e2e recheck conditions).
 * Loads sofc3d_m2.mph (coarse mesh 38980 el, Tmax mesh-insensitive), solves
 * steady std1 for each case, exports full T field for post-processing.
 */
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;

public class EmsRecheck3D {
    static Model model;
    static final String BASE = ProjectPaths.resolve("outputs/verify3d/sofc3d_m2.mph");
    static final String OUT  = ProjectPaths.resolve("outputs/verify3d/out/");
    static final String[] EXPRS = {"x","y","z","dom","T"};
    static final String[][] CASES = {
        {"m15_floor", "0.9575", "1123.0", "6.0", "0.6542"},
        {"mon_hot",   "0.8497", "1123.0", "6.0", "0.7200"},
        {"s5_like",   "0.8000", "1123.0", "6.0", "0.6000"}
    };
    static void p(String s) { System.out.println(s); System.out.flush(); }

    public static void main(String[] args) throws Exception {
        p("EMS_RECHECK3D_START");
        ModelUtil.initStandalone(false);
        long tLoad = System.currentTimeMillis();
        model = ModelUtil.load("Model", BASE);
        p("loaded in " + (System.currentTimeMillis()-tLoad)/1000.0 + "s");
        model.study("std1").setGenPlots(false);
        model.study("std1").setGenIntermediatePlots(false);
        model.study("std2").setGenPlots(false);
        model.study("std2").setGenIntermediatePlots(false);
        model.param().set("V_step_amp", "0[V]");   // steady: no step
        for (String[] c : CASES) {
            String tag = c[0];
            try {
                model.param().set("V_cell", c[1] + "[V]");
                model.param().set("T_in", c[2] + "[K]");
                model.param().set("lambda_air", c[3]);
                model.param().set("FU", c[4]);
                p("== " + tag + " V_cell=" + c[1] + " T_in=" + c[2] + " lam=" + c[3] + " FU=" + c[4] + " ==");
                long t0 = System.currentTimeMillis();
                model.study("std1").run();
                double dt = (System.currentTimeMillis()-t0)/1000.0;
                p("  solve done in " + dt + "s");
                String fname = OUT + "e3d_" + tag + ".csv";
                try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
                model.result().export("data1").set("data", "dset1");
                model.result().export("data1").set("solnum", 1);
                model.result().export("data1").set("expr", EXPRS);
                model.result().export("data1").set("filename", fname);
                model.result().export("data1").run();
                p("  exported " + fname);
            } catch (Throwable t) {
                p("  CASE " + tag + " FAIL: " + String.valueOf(t.getMessage()).split("\n")[0]);
            }
        }
        p("EMS_RECHECK3D_DONE");
        System.exit(0);
    }
}
