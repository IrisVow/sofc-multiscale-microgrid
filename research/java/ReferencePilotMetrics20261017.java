import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class ReferencePilotMetrics20261017{
 static Model m;
 static double eval(String tag,String expr,int[] ds,int sol){m.result().numerical(tag).selection().set(ds);m.result().numerical(tag).set("solnum",sol);m.result().numerical(tag).set("expr",new String[]{expr});return m.result().numerical(tag).getReal()[0][0];}
 public static void main(String[] a){try{
  ModelUtil.initStandalone(false);ModelUtil.showPlots(false);m=ModelUtil.load("PilotMetrics",a[0]);
  for(String type:new String[]{"IntSurface","MaxSurface","MinSurface"}){m.result().numerical().create("pm"+type,type);m.result().numerical("pm"+type).set("data","refTimeData");}
  PrintWriter p=new PrintWriter(a[1],"UTF-8");p.println("solnum,metric,value");
  for(int sol=1;sol<=4;sol++){
   p.printf(Locale.US,"%d,Tmax,%.17g%n",sol,eval("pmMaxSurface","T",new int[]{1,2,3,4,5},sol));
   p.printf(Locale.US,"%d,max_abs_T_change_K,%.17g%n",sol,eval("pmMaxSurface","abs(T-withsol('sol1',T))",new int[]{1,2,3,4,5},sol));
   p.printf(Locale.US,"%d,max_abs_bA_change,%.17g%n",sol,eval("pmMaxSurface","abs(bA-withsol('sol1',bA))",new int[]{1,2,4,5},sol));
   p.printf(Locale.US,"%d,min_bA,%.17g%n",sol,eval("pmMinSurface","bA",new int[]{1,2,4,5},sol));
   p.printf(Locale.US,"%d,max_bA,%.17g%n",sol,eval("pmMaxSurface","bA",new int[]{1,2,4,5},sol));
   for(int d=1;d<=5;d++){
    p.printf(Locale.US,"%d,domain%d_heat_storage_W_per_m,%.17g%n",sol,d,eval("pmIntSurface","ht.C_eff*ht.timeDerivative",new int[]{d},sol));
    if(d!=3){p.printf(Locale.US,"%d,domain%d_mass_storage_kg_per_ms,%.17g%n",sol,d,eval("pmIntSurface","d(fp.epsilon_p*fp.rho,t)",new int[]{d},sol));p.printf(Locale.US,"%d,domain%d_species_storage_kg_per_ms,%.17g%n",sol,d,eval("pmIntSurface","d(fp.epsilon_p*fp.rho*bA,t)",new int[]{d},sol));}
   }
  }p.close();System.out.println("PILOT_METRICS_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
