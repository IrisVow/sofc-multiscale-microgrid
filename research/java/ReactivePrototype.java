import com.comsol.model.*;import com.comsol.model.util.*;import com.comsol.model.physics.*;import java.util.*;
public class ReactivePrototype{
 static void dump(PhysicsFeature f,String p){
  System.out.println("FEATURE|"+p+"|"+f.getType()+"|active="+f.isActive());
  for(String k:f.properties())try{System.out.println(p+"|"+k+"|"+Arrays.deepToString(f.getStringMatrix(k))+"|allowed="+Arrays.toString(f.getAllowedPropertyValues(k)));}catch(Throwable e){try{System.out.println(p+"|"+k+"|"+Arrays.toString(f.getStringArray(k)));}catch(Throwable ee){}}
 }
 public static void main(String[] a)throws Exception{
  ModelUtil.initStandalone(false);ModelUtil.showProgress(false);Model m=ModelUtil.load("ReactivePrototype",a[0]);
  System.out.println("COMPRESSIBILITY_OPTIONS|"+Arrays.toString(m.physics("fp").prop("PhysicalModelProperty").getAllowedPropertyValues("Compressibility")));
  for(String type:new String[]{"Reactions","ReactionSources","MassSource"}){
   String ph=type.equals("MassSource")?"fp":"tcs",tag="proto"+type;
   try{m.physics(ph).create(tag,type,2);m.physics(ph).feature(tag).selection().set(new int[]{2,4});dump(m.physics(ph).feature(tag),ph+"/"+tag);m.physics(ph).feature().remove(tag);}catch(Throwable e){System.out.println("CREATE_ERROR|"+ph+"|"+type+"|"+e.getMessage());}
  }
  for(String type:new String[]{"ReactingFlow","ReactingFlowCoupling","NonisothermalReactingFlow"}){
   String tag="protoCoupling";
   try{m.component("comp1").multiphysics().create(tag,type);System.out.println("COUPLING_TYPE|"+type);
    for(String k:m.component("comp1").multiphysics(tag).properties())try{System.out.println("COUPLING_PROPERTY|"+k+"|"+Arrays.toString(m.component("comp1").multiphysics(tag).getStringArray(k))+"|allowed="+Arrays.toString(m.component("comp1").multiphysics(tag).getAllowedPropertyValues(k)));}catch(Throwable ee){}
    m.component("comp1").multiphysics().remove(tag);
   }catch(Throwable e){System.out.println("COUPLING_CREATE_ERROR|"+type+"|"+e.getMessage());}
  }
  ModelUtil.remove("ReactivePrototype");System.out.println("PROTOTYPE_NO_SOLVE_NO_SAVE");System.exit(0);
 }
}
