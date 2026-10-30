import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class MCSpeciesQuick extends ReactiveExport{
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);m=ModelUtil.load("MCQuick",a[0]);data="dset1";sol=1;cid=a[2];order=8;
 m.result().numerical().create("clArea","IntSurface");m.result().numerical().create("clLine","IntLine");summary(a[1]);w=new PrintWriter(a[1]+"_quick.csv","UTF-8");w.println("case,solnum,quadrature_order,kind,domain,boundary,metric,unit,value,status");
 for(int d:new int[]{1,2,4,5}){record("volume",d,0,"mass_source","kg/(m*s)","fp.Qm");record("volume",d,0,"shape_u","m2","shapeorder(u)");record("volume",d,0,"shape_p","m2","shapeorder(p)");record("volume",d,0,"shape_w1","m2","shapeorder(w1)");for(String sp:new String[]{"w1","w2","w3","w4"})record("volume",d,0,"source_"+sp,"kg/(m*s)","tcs.R_"+sp);}
 for(int b:new int[]{1,9,12,16})for(String sp:new String[]{"w1","w2","w3","w4"})record("boundary",b==1||b==12?1:5,b,"port_"+sp,"kg/(m*s)","tcs.ntflux_"+sp);
 w.close();System.out.println("QUICK_EXPORT_COMPLETE");System.exit(0);
 }catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
