/* DiagExport3D - solve V=0.8 and export fields incl. heat source/current for 2D-vs-3D diagnosis */
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;

public class DiagExport3D {
    static Model model;
    static void p(String s) { System.out.println(s); System.out.flush(); }
    public static void main(String[] args) throws Exception {
        ModelUtil.initStandalone(false);
        String BASE = ProjectPaths.resolve("outputs/verify3d/sofc3d_m2.mph");
        String OUT  = ProjectPaths.resolve("outputs/verify3d/out/");
        String[] EXPRS = {"x","y","z","dom","T","ht.Q","cd.iloc_pcar1","cd.iloc_panr1","phis","phil","u","v","w1","w2","w3"};
        model = ModelUtil.load("Model", BASE);
        model.study("std1").setGenPlots(false);
        model.study("std1").setGenIntermediatePlots(false);
        model.study("std2").setGenPlots(false);
        model.study("std2").setGenIntermediatePlots(false);
        model.param().set("V_cell", "0.8[V]");
        long t0 = System.currentTimeMillis();
        model.study("std1").run();
        p("solve done in " + (System.currentTimeMillis()-t0)/1000 + "s");
        try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
        model.result().export("data1").set("data", "dset1");
        model.result().export("data1").set("solnum", 1);
        model.result().export("data1").set("expr", EXPRS);
        model.result().export("data1").set("filename", OUT + "diag_S5.csv");
        model.result().export("data1").run();
        p("exported diag_S5.csv");
        model.save(OUT + "sofc3d_diag.mph");
        p("DONE " + (System.currentTimeMillis()-t0)/1000 + "s");
        System.exit(0);
    }
}
