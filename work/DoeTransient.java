import com.comsol.model.*;
import com.comsol.model.util.*;

public class DoeTransient {
    // cid, V, T_in, lambda, FU, stepT(s), stepAmp(V)
    static String[][] CASES = {
        {"T0","1.0","1073","8","0.6","100","0.1"},
        {"T1","1.0","1073","8","0.6","50","0.1"},
        {"T2","1.0","1073","8","0.6","200","0.1"},
        {"T3","1.0","1073","8","0.6","100","0.2"}
    };
    public static void main(String[] args) throws Exception {
        DoeCommon.init();
        System.out.println("DOE TRANSIENT START, cases=" + CASES.length);
        for (String[] c : CASES) {
            long t0 = System.currentTimeMillis();
            String cid = c[0], V = c[1], Tin = c[2], lam = c[3], fu = c[4];
            int stepT = Integer.parseInt(c[5]);
            String stepAmp = c[6];
            try {
                DoeCommon.setParams(V, Tin, lam, fu);
                DoeCommon.setSweep(V);
                model().param().set("V_step_amp", stepAmp + "[V]");
                try { model().func("step_load").set("location", stepT + "[s]"); }
                catch (Throwable t) { System.out.println("  step location ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
                // 1) steady base state
                DoeCommon.solveSteady();
                String steadyFile = DoeCommon.OUT + "transient/" + cid + "/sofc1_doe_" + cid + "_steady_" + stepAmp.replace(".","p") + "V.csv";
                java.io.File dir = new java.io.File(DoeCommon.OUT + "transient/" + cid);
                dir.mkdirs();
                DoeCommon.exportSnap("dset1", 1, steadyFile);
                double tm0 = DoeCommon.tmax("dset1", 1);
                // 2) transient
                String[] tlist = DoeCommon.makeTlist(stepT);
                DoeCommon.solveTransient(stepT);
                int n = tlist.length;
                double tmLast = Double.NaN;
                for (int i = 0; i < n; i++) {
                    String tf = DoeCommon.OUT + "transient/" + cid + "/sofc1_doe_" + cid + "_step" + stepT + "_a" + stepAmp.replace(".","p") + "_t" + String.format("%04d", Integer.parseInt(tlist[i])) + "s.csv";
                    DoeCommon.exportSnap("dset2", i + 1, tf);
                    tmLast = DoeCommon.tmax("dset2", i + 1);
                }
                System.out.println("CASE|" + cid + "|V=" + V + "|T_in=" + Tin + "|lambda=" + lam + "|FU=" + fu
                    + "|stepT=" + stepT + "|stepAmp=" + stepAmp + "|nt=" + n + "|Tmax0=" + tm0 + "|TmaxEnd=" + tmLast
                    + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0) + "|OK");
            } catch (Throwable t) {
                System.out.println("CASE|" + cid + "|V=" + V + "|T_in=" + Tin + "|lambda=" + lam + "|FU=" + fu
                    + "|stepT=" + stepT + "|stepAmp=" + stepAmp
                    + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0) + "|FAIL: " + String.valueOf(t.getMessage()).split("\n")[0]);
            }
        }
        System.out.println("DOE TRANSIENT DONE");
    }
    static Model model() { return DoeCommon.model; }
}