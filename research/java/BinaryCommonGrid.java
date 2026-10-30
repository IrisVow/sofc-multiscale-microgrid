import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class BinaryCommonGrid extends ClosureExport{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);m=ModelUtil.load("BinaryCommonGrid",a[0]);m.result().numerical().create("bgProbe","Interp");m.result().numerical("bgProbe").set("data","dset1");m.result().numerical("bgProbe").set("solnum",1);PrintWriter p=new PrintWriter(a[1],"UTF-8");p.println("case,domain,x,y,metric,value");
  for(int d=1;d<=5;d++){int n=401*9;double[][] xy=new double[2][n];for(int i=0;i<401;i++)for(int j=0;j<9;j++){xy[0][i*9+j]=.1*(i+.5)/401;xy[1][i*9+j]=Y[d-1]+(Y[d]-Y[d-1])*(j+.5)/9;}m.result().numerical("bgProbe").selection().set(new int[]{d});m.result().numerical("bgProbe").set("coord",xy);String[] es=d==3?new String[]{"T","phil"}:new String[]{"T","u","v","p","bt_H2","bt_H2O","bt_O2","bt_N2"};for(String e:es){m.result().numerical("bgProbe").set("expr",new String[]{e});double[] val=m.result().numerical("bgProbe").getData()[0][0];for(int j=0;j<n;j++)p.printf(Locale.US,"%s,%d,%.17g,%.17g,%s,%.17g%n",a[2],d,xy[0][j],xy[1][j],e,val[j]);p.flush();}}
  p.close();System.out.println("COMMON_GRID_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
