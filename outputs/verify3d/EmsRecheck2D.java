import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;

public class EmsRecheck2D {
    static final String MODEL = ProjectPaths.resolve("work/sofc1_FINAL2_solved.mph");
    static final String DATASET = "dset3";   // std1 steady solve lands in sol3/dset3
    static final String[][] CASES = {
        {"m15_ep1_peak", "0.957521", "1123.0", "6.0", "0.654167"},
        {"m15_ep1_tail", "0.957510", "1123.0", "6.0", "0.654167"},
        {"m15_ep2_peak", "0.957502", "1123.0", "6.0", "0.654167"},
        {"m15_ep2_tail", "0.957491", "1123.0", "6.0", "0.654167"},
        {"m15_ep3_peak", "0.957483", "1123.0", "6.0", "0.654167"},
        {"m15_ep3_tail", "0.957472", "1123.0", "6.0", "0.654167"},
        {"m15_ep4_peak", "0.957464", "1123.0", "6.0", "0.654167"},
        {"m15_ep4_tail", "0.957453", "1123.0", "6.0", "0.654167"},
        {"m15_ep5_peak", "0.957445", "1123.0", "6.0", "0.654167"},
        {"m15_ep5_tail", "0.957434", "1123.0", "6.0", "0.654167"},
        {"m15_ep6_peak", "0.957428", "1123.0", "6.0", "0.654167"},
        {"m15_ep6_tail", "0.957415", "1123.0", "6.0", "0.654167"},
        {"m15_ep7_peak", "0.957410", "1123.0", "6.0", "0.654167"},
        {"m15_ep7_tail", "0.957399", "1123.0", "6.0", "0.654167"},
        {"mon_hot1", "0.849694", "1123.0", "6.0", "0.720000"},
        {"mon_hot2", "0.849694", "1123.0", "6.0", "0.720000"}
    };
    static void p(String s) { System.out.println(s); System.out.flush(); }
    static final String[][] SCALE = {
        {"comp1_p","1e5"},{"comp1_phil","1"},{"comp1_phis","1"},{"comp1_T","1e3"},
        {"comp1_u","10"},{"comp1_w1","1"},{"comp1_w2","1"},{"comp1_w3","1"}
    };
    static void applyScale(Model model) {
        for (String[] s : SCALE) {
            try {
                model.sol("sol1").feature("v1").feature(s[0]).set("scalemethod", "manual");
                model.sol("sol1").feature("v1").feature(s[0]).set("scaleval", Double.parseDouble(s[1]));
            } catch (Throwable t) {}
        }
    }
    static double tmaxOf(Model model) {
        try {
            model.result().numerical("mxT").set("data", DATASET);
            model.result().numerical("mxT").selection().set(new int[]{1,2,3,4,5});
            model.result().numerical("mxT").set("solnum", 1);
            model.result().numerical("mxT").setResult();
            double[][] v = model.result().numerical("mxT").getReal();
            return (v.length > 0 && v[0].length > 0) ? v[0][0] : Double.NaN;
        } catch (Throwable t) { return Double.NaN; }
    }
    public static void main(String[] args) throws Exception {
        p("EMS_RECHECK_START");
        ModelUtil.initStandalone(false);
        Model model = ModelUtil.load("Model", MODEL);
        try { model.result().numerical().create("mxT", "MaxSurface"); } catch (Throwable t) {}
        model.result().numerical("mxT").set("expr", new String[]{"T"});
        Map<String,Double> cache = new HashMap<>();
        for (String[] c : CASES) {
            String key = c[1]+"|"+c[2]+"|"+c[3]+"|"+c[4];
            if (cache.containsKey(key)) {
                p("CASE|"+c[0]+"|V="+c[1]+"|T_in="+c[2]+"|lam="+c[3]+"|FU="+c[4]+"|Tmax="+cache.get(key)+"|CACHED");
                continue;
            }
            try {
                model.param().set("V_cell", c[1] + "[V]");
                model.param().set("T_in", c[2] + "[K]");
                model.param().set("lambda_air", c[3]);
                model.param().set("FU", c[4]);
                try { model.study("std1").feature("stat").set("plist", new String[]{c[1]}); }
                catch (Throwable t) { p("  plist ERR: " + String.valueOf(t.getMessage()).split("\n")[0]); }
                applyScale(model);
                long t0 = System.currentTimeMillis();
                model.study("std1").run();
                applyScale(model);
                double tm = tmaxOf(model);
                double dt = (System.currentTimeMillis()-t0)/1000.0;
                cache.put(key, tm);
                p("CASE|"+c[0]+"|V="+c[1]+"|T_in="+c[2]+"|lam="+c[3]+"|FU="+c[4]+"|Tmax="+tm+"|time_s="+dt+"|OK");
            } catch (Throwable t) {
                p("CASE|"+c[0]+"|V="+c[1]+"|T_in="+c[2]+"|lam="+c[3]+"|FU="+c[4]+"|FAIL: " + String.valueOf(t.getMessage()).split("\n")[0]);
            }
        }
        p("EMS_RECHECK_DONE");
        System.exit(0);
    }
}
