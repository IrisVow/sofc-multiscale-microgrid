import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class LocalMeshBaseline20261017 {
 public static void main(String[] a){try{
 ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("LocalBaseline",a[0]);
 double[][] xy=m.mesh("mesh1").getVertex();
 m.result().numerical().create("lmProbe","Interp");m.result().numerical("lmProbe").set("data","dset1");m.result().numerical("lmProbe").set("solnum",1);m.result().numerical("lmProbe").set("coord",xy);m.result().numerical("lmProbe").set("expr",new String[]{"T"});double[] t=m.result().numerical("lmProbe").getData()[0][0];
 PrintWriter p=new PrintWriter(a[1],"UTF-8");p.println("case,x,y,T_K");double mx=-1;int im=-1;
 for(int i=0;i<xy[0].length;i++){p.printf(Locale.US,"%s,%.17g,%.17g,%.17g%n",a[2],xy[0][i],xy[1][i],t[i]);if(t[i]>mx){mx=t[i];im=i;}}p.close();
 System.out.printf(Locale.US,"BASELINE|vertices=%d|max_vertex_T=%.17g|x=%.17g|y=%.17g%n",xy[0].length,mx,xy[0][im],xy[1][im]);
 for(String tag:m.mesh("mesh1").feature().tags())System.out.println("MESH_FEATURE|"+tag+"|"+m.mesh("mesh1").feature(tag).getType());
 System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
