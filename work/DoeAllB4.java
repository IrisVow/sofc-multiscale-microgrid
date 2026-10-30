import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.io.File;

/** B4: transient DOE extension -- larger V steps (0.15/0.20 V) x stepT {50,100,200}.
 *  Replicates DoeAll.runTransient machinery; one case per JVM: DoeAllB4 <tag>.
 */
public class DoeAllB4 {
    static Model model;
    static final String MODEL = ProjectPaths.resolve("work/sofc1_FINAL2_solved.mph");
    static final String OUT = ProjectPaths.resolve("work/doe_b4/transient/");
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
    // tag -> (V_cell, T_in, lambda_air, FU, stepT, stepAmp)
    static final Map<String, String[]> CASES = new LinkedHashMap<>();
    static {
        CASES.put("T4", new String[]{"1.0","1073","8","0.6","100","0.15"});
        CASES.put("T5", new String[]{"1.0","1073","8","0.6","200","0.15"});
        CASES.put("T6", new String[]{"1.0","1073","8","0.6","50","0.15"});
        CASES.put("T7", new String[]{"1.0","1073","8","0.6","100","0.20"});
        CASES.put("T8", new String[]{"1.0","1073","8","0.6","200","0.20"});
        CASES.put("T9", new String[]{"1.0","1073","8","0.6","50","0.20"});
    }

    public static void main(String[] args) throws Exception {
        ModelUtil.initStandalone(false);
        if (args.length == 0) {
            for (Map.Entry<String, String[]> e : CASES.entrySet()) {
                String[] c = e.getValue();
                System.out.println(e.getKey() + " V=" + c[0] + " T=" + c[1] + " lam=" + c[2] + " FU=" + c[3]
                        + " stepT=" + c[4] + " amp=" + c[5]);
            }
            System.exit(0);
        }
        String tag = args[0];
        new File(OUT + tag).mkdirs();
        String[] c = CASES.get(tag);
        if (c == null) { System.err.println("unknown tag: " + tag); System.exit(2); }
        model = ModelUtil.load("Model", MODEL);
        try { model.result().export().create("data1", "Data"); } catch (Throwable t) {}
        try { model.result().numerical().create("mxT", "MaxSurface"); } catch (Throwable t) {}
        model.result().numerical("mxT").set("expr", new String[]{"T"});
        long t0 = System.currentTimeMillis();
        String cid = tag, V = c[0], Tin = c[1], lam = c[2], fu = c[3];
        int stepT = Integer.parseInt(c[4]);
        String stepAmp = c[5];
        try {
            setParams(V, Tin, lam, fu);
            model.param().set("V_step_amp", stepAmp + "[V]");
            try { model.func("step_load").set("location", stepT + "[s]"); }
            catch (Throwable t) { System.out.println("  step location ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
            solveSteady(V);
            String steadyFile = OUT + cid + "/sofc1_doe_" + cid + "_steady.csv";
            exportSnap("dset1", 1, steadyFile);
            double tm0 = tmax("dset1", 1);
            String[] tlist = makeTlist(stepT);
            solveTransient(stepT);
            double tmLast = Double.NaN;
            for (int i = 0; i < tlist.length; i++) {
                String tf = OUT + cid + "/sofc1_doe_" + cid + "_t" + String.format("%04d", Integer.parseInt(tlist[i])) + "s.csv";
                exportSnap("dset2", i + 1, tf);
                tmLast = tmax("dset2", i + 1);
            }
            System.out.println("CASE|" + cid + "|V=" + V + "|T_in=" + Tin + "|lambda=" + lam + "|FU=" + fu
                    + "|stepT=" + stepT + "|stepAmp=" + stepAmp + "|nt=" + tlist.length + "|Tmax0=" + tm0 + "|TmaxEnd=" + tmLast
                    + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0) + "|OK");
        } catch (Throwable t) {
            System.out.println("CASE|" + cid + "|V=" + V + "|T_in=" + Tin + "|lambda=" + lam + "|FU=" + fu
                    + "|stepT=" + stepT + "|stepAmp=" + stepAmp
                    + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0) + "|FAIL " + String.valueOf(t.getMessage()).split("\n")[0]);
        }
        System.exit(0);
    }

    static void setParams(String V, String Tin, String lam, String fu) {
        model.param().set("V_cell", V + "[V]");
        model.param().set("T_in", Tin + "[K]");
        model.param().set("lambda_air", lam);
        model.param().set("FU", fu);
    }

    static void solveSteady(String V) {
        try { model.sol("sol1").feature("s1").feature("p1").set("plist", new String[]{V}); }
        catch (Throwable t) { System.out.println("  p1 plist ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        applyScale();
        model.sol("sol1").runAll();
        applyScale();
    }

    static void solveTransient(int stepT) {
        model.study("std2").feature("time").set("tlist", makeTlist(stepT));
        try {
            model.sol("sol2").feature("v1").set("initmethod", "sol");
            model.sol("sol2").feature("v1").set("initsol", "sol1");
            model.sol("sol2").feature("v1").set("solnum", 1);
        } catch (Throwable t) { System.out.println("  sol2 init ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        model.sol("sol2").runAll();
    }

    static void applyScale() {
        for (String[] s : SCALE) {
            try {
                model.sol("sol1").feature("v1").feature(s[0]).set("scalemethod", "manual");
                model.sol("sol1").feature("v1").feature(s[0]).set("scaleval", Double.parseDouble(s[1]));
            } catch (Throwable t) { System.out.println("  scale " + s[0] + " ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
        }
    }

    static String[] makeTlist(int stepT) {
        int[] base = {0,10,20,30,40,50,60,80,99,101,110,125,150,180,220,280,360,480,640,840,1200};
        TreeSet<Integer> ts = new TreeSet<>();
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
}

