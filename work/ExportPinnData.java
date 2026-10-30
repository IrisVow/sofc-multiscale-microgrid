import com.comsol.model.*;
import com.comsol.model.util.*;

public class ExportPinnData {
    static String work = ProjectPaths.resolve("work/pinndata/");
    static String modelPath = ProjectPaths.resolve("work/sofc1_FINAL2_solved.mph");

    public static void main(String[] args) throws Exception {
        boolean testOnly = args.length > 0 && args[0].equals("test");
        Model model = ModelUtil.load("Model", modelPath);
        System.out.println("model loaded");

        String[] steadyExprs = {"T","u","v","p","w1","w2","w3","w4","phis","phil","tcs.rho",
            "w1*tcs.rho/M_H2","w2*tcs.rho/M_H2O","w3*tcs.rho/M_O2","w4*tcs.rho/M_N2"};
        String[] transientExprs = new String[steadyExprs.length + 1];
        transientExprs[0] = "t";
        System.arraycopy(steadyExprs, 0, transientExprs, 1, steadyExprs.length);

        try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
        String tag = "mxT";
        try { model.result().numerical().create(tag, "MaxSurface"); } catch (Throwable t) {}
        model.result().numerical(tag).set("expr", new String[]{"T"});

        double[] Vlist = {1.0, 0.95, 0.9, 0.85, 0.8, 0.75};
        String[] Vlabel = {"V1.000","V0.950","V0.900","V0.850","V0.800","V0.750"};

        if (!testOnly) {
            System.out.println("==== STEADY dset1 ====");
            for (int i = 0; i < 6; i++) {
                int sol = i + 1;
                String fname = work + "steady/sofc1_steady_" + Vlabel[i] + "_V.csv";
                try {
                    model.result().export("data1").set("data", "dset1");
                    model.result().export("data1").set("solnum", sol);
                    model.result().export("data1").set("expr", steadyExprs);
                    model.result().export("data1").set("filename", fname);
                    model.result().export("data1").run();
                    System.out.println("steady sol" + sol + " V=" + Vlist[i] + " exported -> " + fname);
                } catch (Throwable t) { System.out.println("steady sol" + sol + " ERR: " + t); }
                try {
                    model.result().numerical(tag).set("data", "dset1");
                    model.result().numerical(tag).selection().set(new int[]{1,2,3,4,5});
                    model.result().numerical(tag).set("solnum", sol);
                    model.result().numerical(tag).setResult();
                    double[][] vals = model.result().numerical(tag).getReal();
                    System.out.println("  Tmax sol" + sol + " = " + (vals.length > 0 && vals[0].length > 0 ? vals[0][0] : Double.NaN));
                } catch (Throwable t) { System.out.println("  Tmax sol" + sol + " ERR: " + t); }
            }
        }

        double[] tl = {0,10,20,30,40,50,60,80,99,101,110,125,150,180,220,280,360,480,640,840,1200};
        int start = testOnly ? 20 : 0;
        int end   = testOnly ? 21 : tl.length;
        System.out.println("==== TRANSIENT dset2 ====");
        for (int i = start; i < end; i++) {
            int sol = i + 1;
            String fname = work + "transient/sofc1_transient_t" + String.format("%04d", (int)tl[i]) + "s.csv";
            try {
                model.result().export("data1").set("data", "dset2");
                model.result().export("data1").set("solnum", sol);
                model.result().export("data1").set("expr", transientExprs);
                model.result().export("data1").set("filename", fname);
                model.result().export("data1").run();
                System.out.println("transient sol" + sol + " t=" + tl[i] + "s exported -> " + fname);
            } catch (Throwable t) { System.out.println("transient sol" + sol + " ERR: " + t); }
            try {
                model.result().numerical(tag).set("data", "dset2");
                model.result().numerical(tag).selection().set(new int[]{1,2,3,4,5});
                model.result().numerical(tag).set("solnum", sol);
                model.result().numerical(tag).setResult();
                double[][] vals = model.result().numerical(tag).getReal();
                System.out.println("  Tmax sol" + sol + " = " + (vals.length > 0 && vals[0].length > 0 ? vals[0][0] : Double.NaN));
            } catch (Throwable t) { System.out.println("  Tmax sol" + sol + " ERR: " + t); }
        }
        System.out.println("DONE");
    }
}