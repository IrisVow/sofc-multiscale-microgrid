import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;
import java.io.File;

/** B4 cont: continuous load trajectories (ramp / multistep / sine / random), V in 0.75-1.0 V, 1200 s window.
 *  Rebuilds step_load as Analytic wave function; one case per JVM: DoeContB4 <tag>.
 */
public class DoeContB4 {
    static Model model;
    static final String MODEL = ProjectPaths.resolve("work/sofc1_FINAL2_solved.mph");
    static final String OUT = ProjectPaths.resolve("work/doe_cont/transient/");
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
    // tag -> (waveExpr, V_step_amp_V)
    static final Map<String, String[]> CASES = new LinkedHashMap<>();
    static {
        CASES.put("R1", new String[]{"min(t/600, 1)", "0.2"});
        CASES.put("R2", new String[]{"if(t<200,0,if(t<400,0.2,if(t<600,0.6,if(t<800,1.0,if(t<1000,0.4,0.8)))))", "0.25"});
        CASES.put("R3", new String[]{"0.5+0.5*sin(2*pi*t/600)", "0.2"});
        CASES.put("R4", new String[]{"if(t<150,0,if(t<300,0.6,if(t<500,0.2,if(t<700,1.0,if(t<900,0.4,if(t<1100,0.8,0.6))))))", "0.25"});
    }

    public static void main(String[] args) throws Exception {
        ModelUtil.initStandalone(false);
        if (args.length == 0) {
            for (Map.Entry<String, String[]> e : CASES.entrySet()) {
                System.out.println(e.getKey() + " wave=" + e.getValue()[0] + " amp=" + e.getValue()[1]);
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
        try { model.result().numerical("mxT").set("expr", new String[]{"T"}); } catch (Throwable t) {}
        long t0 = System.currentTimeMillis();
        try {
            String wave = c[0], amp = c[1];
            rebuildStepLoad(wave);
            model.param().set("V_cell", "1.0[V]");
            model.param().set("V_step_amp", amp + "[V]");
            model.param().set("T_in", "1073[K]");
            model.param().set("lambda_air", "8");
            model.param().set("FU", "0.6");
            solveSteady();
            String steadyFile = OUT + tag + "/sofc1_doe_" + tag + "_steady.csv";
            exportSnap("dset1", 1, steadyFile);
            double tm0 = tmax("dset1", 1);
            String[] tlist = makeTlist(tag);
            solveTransient(tlist);
            double tmLast = Double.NaN;
            for (int i = 0; i < tlist.length; i++) {
                String tf = OUT + tag + "/sofc1_doe_" + tag + "_t" + String.format("%04d", Integer.parseInt(tlist[i])) + "s.csv";
                exportSnap("dset2", i + 1, tf);
                tmLast = tmax("dset2", i + 1);
            }
            System.out.println("CASE|" + tag + "|wave=" + wave + "|amp=" + amp + "|nt=" + tlist.length
                    + "|Tmax0=" + tm0 + "|TmaxEnd=" + tmLast
                    + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0) + "|OK");
        } catch (Throwable t) {
            System.out.println("CASE|" + tag + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0)
                    + "|FAIL " + String.valueOf(t.getMessage()).split("\n")[0]);
        }
        System.exit(0);
    }

    static void rebuildStepLoad(String expr) {
        try { model.func().remove("step_load"); } catch (Throwable t) { System.out.println("  remove ERR: " + msg(t)); }
        model.func().create("step_load", "Analytic");
        model.func("step_load").set("expr", expr);
        model.func("step_load").set("args", new String[]{"t"});
        try { model.func("step_load").set("funcname", "step_load"); } catch (Throwable t) { System.out.println("  funcname ERR: " + msg(t)); }
        System.out.println("  step_load -> " + expr);
    }

    static void solveSteady() {
        try { model.sol("sol1").feature("s1").feature("p1").set("plist", new String[]{"1.0"}); }
        catch (Throwable t) { System.out.println("  p1 plist ERR: " + msg(t)); }
        applyScale();
        model.sol("sol1").runAll();
        applyScale();
    }

    static void solveTransient(String[] tlist) {
        model.study("std2").feature("time").set("tlist", tlist);
        try {
            model.sol("sol2").feature("v1").set("initmethod", "sol");
            model.sol("sol2").feature("v1").set("initsol", "sol1");
            model.sol("sol2").feature("v1").set("solnum", 1);
        } catch (Throwable t) { System.out.println("  sol2 init ERR: " + msg(t)); }
        model.sol("sol2").runAll();
    }

    static void applyScale() {
        for (String[] s : SCALE) {
            try {
                model.sol("sol1").feature("v1").feature(s[0]).set("scalemethod", "manual");
                model.sol("sol1").feature("v1").feature(s[0]).set("scaleval", Double.parseDouble(s[1]));
            } catch (Throwable t) { System.out.println("  scale " + s[0] + " ERR: " + msg(t)); }
        }
    }

    static String[] makeTlist(String tag) {
        TreeSet<Integer> ts = new TreeSet<>();
        if (tag.equals("R1")) {
            int[] a = {0,10,20,30,40,50,60,80,100,125,150,180,220,280,360,480,600,620,650,700,800,900,1000,1100,1200};
            for (int x : a) ts.add(x);
        } else if (tag.equals("R2")) {
            int[] a = {0,10,20,30,40,50,60,80,100,125,150,180,200,220,250,280,320,360,400,420,450,500,550,600,620,650,700,750,800,820,850,900,950,1000,1020,1050,1100,1150,1200};
            for (int x : a) ts.add(x);
        } else if (tag.equals("R3")) {
            int[] a = {0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200};
            for (int x : a) ts.add(x);
        } else {
            int[] a = {0,10,20,30,40,50,60,80,100,125,150,160,175,200,225,250,275,300,325,350,400,450,500,510,525,550,600,650,700,710,725,750,800,850,900,910,925,950,1000,1050,1100,1110,1125,1150,1200};
            for (int x : a) ts.add(x);
        }
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

    static String msg(Throwable t) { return String.valueOf(t.getMessage()).split("\n")[0]; }
}