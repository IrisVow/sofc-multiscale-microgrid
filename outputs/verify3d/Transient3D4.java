/* Transient3D4 - replicate 2D GUI sol2 config (events + tight rtol/atol + fully coupled).
 * Usage: Transient3D4 <amp> <stepT> <prefix>
 * Loads sofc3d_solved_steady.mph (sol1 = V=1.0 steady), runs sol2.runAll() with step-load BC.
 */
import com.comsol.model.*;
import com.comsol.model.util.*;

public class Transient3D4 {
    static Model m;
    static void p(String s) { System.out.println(s); System.out.flush(); }
    static final String DIR = ProjectPaths.resolve("outputs/verify3d/");
    static final String OUT = DIR + "out/";
    static final String STEADY = DIR + "sofc3d_solved_steady.mph";
    static final String TRAN = DIR + "sofc3d_solved_transient4.mph";
    static final String[] TLIST = {"0","10","20","30","40","50","60","80","99","100","101","110","125","150","180","220","280","360","480","640","840","1200"};
    static final String[] SMOKE = {"0","60","80","99","100","101","110","150"};

    public static void main(String[] args) throws Exception {
        ModelUtil.initStandalone(false);
        String amp = args.length > 0 ? args[0] : "0.2";
        String stepT = args.length > 1 ? args[1] : "100";
        String prefix = args.length > 2 ? args[2] : "t4";
        String mode = args.length > 3 ? args[3] : "full";
        m = ModelUtil.load("Model", STEADY);
        m.study("std1").setGenPlots(false); m.study("std1").setGenIntermediatePlots(false);
        m.study("std2").setGenPlots(false); m.study("std2").setGenIntermediatePlots(false);
        m.param().set("V_cell", "1.0[V]");
        m.param().set("V_step_amp", amp + "[V]");
        try { m.func("step_load").set("location", stepT + "[s]"); } catch (Throwable t) { p("step loc ERR " + t); }
        try { m.result().export().create("data1", "Data"); } catch (Throwable t) {}

        m.sol().create("sol2");
        com.comsol.model.SolverSequence sq = m.sol("sol2");
        sq.study("std2");
        sq.createAutoSequence("std2");
        // event handling for the load step (mirror 2D GUI sol2)
        try { sq.feature("st1").set("useent", "all"); p("st1 useent=all"); }
        catch (Throwable t) { p("st1 ERR " + String.valueOf(t.getMessage()).split("\n")[0]); }
        com.comsol.model.SolverFeature t1 = sq.feature("t1");
        try { t1.feature("se1").active(false); p("se1 disabled"); } catch (Throwable t) { p("se1 ERR " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            com.comsol.model.SolverFeature fc1 = t1.create("fc1", "FullyCoupled");
            fc1.set("linsolver", "dDef");
            fc1.set("initstep", "1");
            fc1.set("minstep", "1e-4");
            fc1.set("maxiter", "300");
            fc1.set("jtech", "once");
            t1.feature("dDef").set("linsolver", "pardiso");
            p("fc1 added (maxiter=300)");
        } catch (Throwable t) { p("fc1 ERR " + String.valueOf(t.getMessage()).split("\n")[0]); }
        // time solver settings mirroring 2D GUI
        try {
            t1.set("rtol", "0.001");
            t1.set("atolglobalmethod", "scaled");
            t1.set("atolglobalvaluemethod", "factor");
            t1.set("atolglobalfactor", "0.05");
            t1.set("eventout", "on");
            t1.set("eventtol", "0.01");
            t1.set("maxorder", "2");
            t1.set("minorder", "1");
            t1.set("stabcntrl", "on");
            t1.set("masssingular", "maybe");
            t1.set("consistent", "bweuler");
            t1.set("tstepsbdf", "free");
            t1.set("endtimeinterpolation", "on");
            t1.set("tstepsstore", "1");
            p("t1 settings applied");
        } catch (Throwable t) { p("t1 set ERR " + String.valueOf(t.getMessage()).split("\n")[0]); }
        try {
            sq.feature("v1").set("initmethod", "sol");
            sq.feature("v1").set("initsol", "sol1");
            sq.feature("v1").set("solnum", 1);
            p("v1 init from sol1");
        } catch (Throwable t) { p("v1 ERR " + String.valueOf(t.getMessage()).split("\n")[0]); }
        String[][] SCALE = {
            {"comp1_p","1e5"},{"comp1_phil","1"},{"comp1_phis","1"},{"comp1_T","1e3"},
            {"comp1_u","10"},{"comp1_w2","1"},{"comp1_w3","1"},{"comp1_w4","1"}
        };
        for (String[] s : SCALE) {
            try {
                com.comsol.model.SolverFeature f = sq.feature("v1").feature(s[0]);
                f.set("scalemethod", "manual");
                f.set("scaleval", Double.parseDouble(s[1]));
            } catch (Throwable t) { p("scale " + s[0] + " ERR " + String.valueOf(t.getMessage()).split("\n")[0]); }
        }
        String[] tlist = mode.equals("smoke") ? SMOKE : TLIST;
        m.study("std2").feature("time").set("tlist", String.join(" ", tlist));
        p("== run sol2.runAll() (" + tlist.length + " times, rtol=1e-3, events on) ==");
        long t0 = System.currentTimeMillis();
        try { sq.runAll(); } catch (Throwable t) { p("runAll THREW " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("sol2 done " + (System.currentTimeMillis()-t0)/1000 + "s");
        for (int i = 0; i < tlist.length; i++) {
            String fname = OUT + prefix + "_t" + tlist[i] + "s.csv";
            try {
                m.result().export("data1").set("data", "dset2");
                m.result().export("data1").set("solnum", i + 1);
                m.result().export("data1").set("expr", new String[]{"x","y","z","dom","T","phis","ht.Q"});
                m.result().export("data1").set("filename", fname);
                m.result().export("data1").run();
                p("  exported " + fname);
            } catch (Throwable t) { p("  export ERR sn=" + (i+1) + " " + String.valueOf(t.getMessage()).split("\n")[0]); }
        }
        try { m.save(TRAN); p("saved " + TRAN); } catch (Throwable t) { p("save ERR " + String.valueOf(t.getMessage()).split("\n")[0]); }
        p("DONE total=" + (System.currentTimeMillis()-t0)/1000 + "s");
        System.exit(0);
    }
}
