import com.comsol.model.*;import com.comsol.model.util.*;import java.io.*;import java.util.*;
public class LocalMeshQuality20261017 {
 public static void export(Model m,String path)throws Exception{
  m.mesh("mesh1").stat().selection().geom(2);m.mesh("mesh1").stat().selection().set(new int[]{1,2,3,4,5});
  PrintWriter p=new PrintWriter(path,"UTF-8");p.println("element_type,elements,quality_measure,min_quality,mean_quality,min_area_m2,max_area_m2,total_area_m2");
  for(String type:new String[]{"tri","quad"}){int n=m.mesh("mesh1").stat().getNumElem(type);if(n>0)p.printf(Locale.US,"%s,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g%n",type,n,m.mesh("mesh1").stat().getQualityMeasure(),m.mesh("mesh1").stat().getMinQuality(type),m.mesh("mesh1").stat().getMeanQuality(type),m.mesh("mesh1").stat().getMinVolume(type),m.mesh("mesh1").stat().getMaxVolume(type),m.mesh("mesh1").stat().getVolume(type));}
  p.close();
 }
 public static void main(String[] a){try{ModelUtil.initStandalone(false);ModelUtil.showPlots(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("LocalQuality",a[0]);export(m,a[1]);System.out.println("MESH_QUALITY_EXPORTED");System.exit(0);}catch(Throwable e){e.printStackTrace(System.out);System.exit(1);}}
}
