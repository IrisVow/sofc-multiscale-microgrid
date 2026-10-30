import com.comsol.model.*;import com.comsol.model.util.*;
public class ReactiveDeliverModel{
 public static void main(String[] a){try{run(a);}catch(Throwable e){System.out.println("DELIVERY_MODEL_FAILED");e.printStackTrace(System.out);System.exit(1);}}
 static void run(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showProgress(false);ModelUtil.showPlots(false);Model m=ModelUtil.load("DeliverReactive",a[0]);
  if(Math.abs(m.param().evaluate("alpha_rxn")-1)>1e-12)throw new RuntimeException("Non-final source ramp");
  for(String tag:m.sol().tags())if(tag.equals("sol2")||tag.equals("sol3")){m.sol(tag).clearSolutionData();m.sol(tag).label("Legacy "+tag+" cleared: not recomputed");}
  for(String tag:m.result().dataset().tags()){
   if(tag.equals("dset1"))m.result().dataset(tag).label("CONVERGED candidate reactive steady "+a[2]+", alpha_rxn=1");
   else if(tag.equals("dset2")||tag.equals("dset3"))m.result().dataset(tag).label("EMPTY legacy "+tag+": not recomputed");
  }
  m.label("Reactive porous SOFC "+a[2]+" - steady validation candidate");m.save(a[1]);ModelUtil.remove("DeliverReactive");System.out.println("DELIVERY_MODEL_SAVED|"+a[2]);System.exit(0);
 }
}
