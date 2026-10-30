import com.comsol.model.*;
import com.comsol.model.util.*;

public class DoeSteady {
    static String[][] CASES = {
        {"S1","1.0","1073","8","0.6"},
        {"S2","0.95","1073","8","0.6"},
        {"S3","0.9","1073","8","0.6"},
        {"S4","0.85","1073","8","0.6"},
        {"S5","0.8","1073","8","0.6"},
        {"S6","0.75","1073","8","0.6"},
        {"S7","0.9","1073","4","0.6"},
        {"S8","0.9","1073","8","0.8"},
        {"S9","0.9","1123","8","0.6"}
    };
    public static void main(String[] args) throws Exception {
        DoeCommon.init();
        System.out.println("DOE STEADY START, cases=" + CASES.length);
        for (String[] c : CASES) {
            long t0 = System.currentTimeMillis();
            String cid = c[0], V = c[1], Tin = c[2], lam = c[3], fu = c[4];
            try {
                DoeCommon.setParams(V, Tin, lam, fu);
                DoeCommon.setSweep(V);
                DoeCommon.solveSteady();
                String fname = DoeCommon.OUT + "steady/sofc1_doe_" + DoeCommon.label(cid, V, Tin, lam, fu) + ".csv";
                DoeCommon.exportSnap("dset1", 1, fname);
                double tm = DoeCommon.tmax("dset1", 1);
                System.out.println("CASE|" + cid + "|V=" + V + "|T_in=" + Tin + "|lambda=" + lam + "|FU=" + fu
                    + "|Tmax=" + tm + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0) + "|OK");
            } catch (Throwable t) {
                System.out.println("CASE|" + cid + "|V=" + V + "|T_in=" + Tin + "|lambda=" + lam + "|FU=" + fu
                    + "|time_s=" + ((System.currentTimeMillis() - t0) / 1000.0) + "|FAIL: " + String.valueOf(t.getMessage()).split("\n")[0]);
            }
        }
        System.out.println("DOE STEADY DONE");
    }
}