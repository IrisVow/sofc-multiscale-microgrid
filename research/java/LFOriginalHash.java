import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;
public class LFOriginalHash{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);PrintWriter p=new PrintWriter(a[1],"UTF-8");p.println("case,original_solution_sha256");for(String key:new String[]{"M0_S5","M1_S5","M1_S6"}){Model m=ModelUtil.load("LFHash",a[0]+"/"+key+"_corrected.mph");p.println(key+","+LFFrozenRecovery.hash(m.sol("sol1").getU()));p.flush();ModelUtil.remove("LFHash");}p.close();System.out.println("ORIGINAL_HASHES_EXPORTED");System.exit(0);}catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
