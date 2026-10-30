import com.comsol.model.*;
import com.comsol.model.util.*;

public class DoeCommon {
    static Model model;
    static final String MODEL = ProjectPaths.resolve("work/sofc1_FINAL2_solved.mph");
    static final String OUT = ProjectPaths.resolve("work/doe/");
    static final String[] EXPRS = {
        "T","u","v","p","w1","w2","w3","w4","phis","phil","tcs.rho",
        "w1*tcs.rho/M_H2","w2*tcs.rho/M_H2O","w3*tcs.rho/M_O2","w4*tcs.rho/M_N2",
        "ht.Q",
        "cd.isx","cd.isy","cd.ilx","cd.ily",
        "cd.iloc_pcar1","cd.iloc_panr1",
        "cd.eta_pcar1","cd.eta_panr1",
        "cd.Eeq_pcar1","cd.Eeq_panr1",
        "cd.ilim_pcar1","cd.ilim_panr1",
        "cd.iloc_panr1*Av_an","cd.iloc_pcar1*Av_ca",
        "-cd.iloc_panr1*Av_an*M_H2/(2*F_const)","cd.iloc_panr1*Av_an*M_H2O/(2*F_const)","-cd.iloc_pcar1*Av_ca*M_O2/(4*F_const)"
    };
    static final String[][] SCALE = {
        {"comp1_p","1e5"},{"comp1_phil","1"},{"comp1_phis","1"},{"comp1_T","1e3"},
        {"comp1_u","10"},{"comp1_w1","1"},{"comp1_w2","1"},{"comp1_w3","1"}
    };

    public static void init() throws Exception {
        model = ModelUtil.load("Model", MODEL);
        try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
        try { model.result().numerical().create("mxT", "MaxSurface"); } catch (Throwable t) {}
        model.result().numerical("mxT").set("expr", new String[]{"T"});
    }

    static void setParams(String V, String Tin, String lam, String fu) {
        model.param().set("V_cell", V + "[V]");
        model.param().set("T_in", Tin + "[K]");
        model.param().set("lambda_air", lam);
        model.param().set("FU", fu);
    }

    static void setSweep(String V) {
        try { model.study("std1").feature("stat").set("plist", new String[]{V}); }
        catch (Throwable t) { System.out.println("  plist ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
    }

    static void applyScale() {
        for (String[] s : SCALE) {
            try {
                model.sol("sol1").feature("v1").feature(s[0]).set("scalemethod", "manual");
                model.sol("sol1").feature("v1").feature(s[0]).set("scaleval", Double.parseDouble(s[1]));
            } catch (Throwable t) { System.out.println("  scale " + s[0] + " ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        }
    }

    static void solveSteady() {
        applyScale();
        model.study("std1").run();
        applyScale();
    }

    static void solveTransient(int stepT) {
        model.study("std2").feature("time").set("tlist", makeTlist(stepT));
        model.study("std2").run();
        try {
            model.sol("sol2").feature("v1").set("initmethod", "sol");
            model.sol("sol2").feature("v1").set("initsol", "sol1");
            model.sol("sol2").feature("v1").set("solnum", 1);
        } catch (Throwable t) { System.out.println("  sol2 init reset ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
    }

    static String[] makeTlist(int stepT) {
        int[] base = {0,10,20,30,40,50,60,80,99,101,110,125,150,180,220,280,360,480,640,840,1200};
        java.util.TreeSet<Integer> ts = new java.util.TreeSet<>();
        for (int b : base) ts.add(b);
        if (stepT > 0) ts.add(stepT);
        String[] out = new String[ts.size()];
        int i = 0;
        for (int t : ts) out[i++] = String.valueOf(t);
        return out;
    }

    static void exportSnap(String dataset, int solnum, String fname) {
        model.result().export("data1").set("data", dataset);
        model.result().export("data1").set("solnum", solnum);
        model.result().export("data1").set("expr", EXPRS);
        model.result().export("data1").set("filename", fname);
        model.result().export("data1").run();
    }

    static double tmax(String dataset, int solnum) {
        try {
            model.result().numerical("mxT").set("data", dataset);
            model.result().numerical("mxT").selection().set(new int[]{1,2,3,4,5});
            model.result().numerical("mxT").set("solnum", solnum);
            model.result().numerical("mxT").setResult();
            double[][] v = model.result().numerical("mxT").getReal();
            return (v.length > 0 && v[0].length > 0) ? v[0][0] : Double.NaN;
        } catch (Throwable t) { return Double.NaN; }
    }

    static String label(String cid, String V, String Tin, String lam, String fu) {
        return cid + "_V" + V + "_T" + Tin + "_L" + lam + "_FU" + fu;
    }
}