import{f as c,B as i,az as x,j as e,r as m,bM as j,aK as v,bi as p,bN as w,bO as M,bP as O,bQ as R,F as g,bR as B,a6 as E,Z as K,bS as I,bE as L,bJ as A,L as W,bT as k,T as N,aZ as $}from"./index.5bdb0a85.js";import F from"./ModelEditorMarkdown.37ff2a5e.js";const T=({children:a,title:r,...n})=>c(i,{sx:{mb:1},children:[c(x,{fontWeight:"bold",children:[r,":"]}),e(i,{...n,children:a})]}),H=({inferenceColumn:a,handleChange:r,value:n})=>{const t=()=>({onChange:r,value:n,label:a.name,key:a.name}),d=m.exports.useCallback(u=>e(j,{...t(),domainKind:u}),[a.dataType]),l=e(v,{color:"error"},a.name);return!a.dataType||!a.dataType.domainKind?l:{[p.Smiles]:e(w,{...t()}),[p.Numerical]:e(M,{...t(),unit:"unit"in a.dataType?a.dataType.unit:"",value:n}),[p.Categorical]:e(O,{...t(),getLabel:u=>u,options:"classes"in a.dataType?Object.keys(a.dataType.classes):[]}),[p.String]:e(R,{...t()}),[p.Dna]:d(a.dataType.domainKind),[p.Rna]:d(a.dataType.domainKind),[p.Protein]:d(a.dataType.domainKind)}[a.dataType.domainKind]||l},Q=({inferenceColumns:a,values:r,handleChange:n})=>e(i,{sx:{mb:"1rem",ml:"5px",border:"1px solid rgba(0, 0, 0, 0.12)",padding:"1rem",borderRadius:"4px"},children:a.map(t=>e(i,{sx:{mb:"1rem"},children:e(H,{inferenceColumn:t,value:r[t.name],handleChange:d=>n(t.name,d)})},t.name))}),Z=({outputValues:a,targetColumns:r})=>c(g,{children:[e(x,{fontWeight:"bold",marginBottom:"0.5rem",marginTop:"2rem",children:"Output:"},"1"),e(i,{sx:{display:"flex",flexWrap:"wrap",flexDirection:"row",gap:"5px",ml:"8px",justifyContent:"space-around"},children:Object.keys(a).map(n=>{const t=r.find(l=>l.name===n);if(!t)return null;const d=t.columnType=="regression"?"numerical":"categorical";return e(i,{sx:{mb:"1rem",mt:"1rem",border:"1px solid rgba(0, 0, 0, 0.12)",padding:"1rem",borderRadius:"4px",maxWidth:"360px",width:"100%",display:"flex",flexDirection:"column",justifyContent:"flex-start",alignItems:"flex-start",height:"140px"},children:e(B,{column:t.name,unit:"unit"in t.dataType?t.dataType.unit:"",type:d,value:a[t.name]})},t.name)})},"2")]}),z=async(a,r)=>(Object.keys(r).forEach(t=>{Array.isArray(r[t])||(r[t]=[r[t]])}),(await I.post(`api/v1/deployments/${a.id}/predict`,r)).data),J=async(a,r)=>(Object.keys(r).forEach(t=>{Array.isArray(r[t])||(r[t]=[r[t]])}),(await I.post(`api/v1/deployments/${a.id}/predict-public`,r)).data),U=({deployment:a,publicDeployment:r})=>{const n=r?J:z,t=m.exports.useMemo(()=>{var s,o;return(o=(s=a.modelVersion)==null?void 0:s.config.dataset)==null?void 0:o.featureColumns},[a.id]),d=m.exports.useMemo(()=>{var s,o;return(o=(s=a.modelVersion)==null?void 0:s.config.dataset)==null?void 0:o.targetColumns},[a.id]);if(!t||!d)return null;const[l,u]=m.exports.useState(t==null?void 0:t.reduce((s,o)=>({...s,[o.name]:""}),{})),D=(s,o)=>u({...l,[s]:o}),[P,h]=m.exports.useState(!1),{setMessage:S}=E(),[b,y]=m.exports.useState(null);return c(i,{children:[c(i,{sx:{mb:1,display:"flex",justifyContent:"space-between",alignItems:"center"},children:[e(x,{fontWeight:"bold",children:"Input:"}),e(K,{onClick:()=>{if(!P){if(h(!0),Object.values(l).some(s=>[null,void 0,""].includes(s)))return y(null);n(a,l).then(y).catch(s=>{var o,f;return((f=(o=s.response)==null?void 0:o.data)==null?void 0:f.detail)&&S({message:s.response.data.detail,type:"error"})}).finally(()=>h(!1))}},variant:"contained",color:"primary",sx:{ml:3},children:"Predict"})]}),e(Q,{inferenceColumns:t,handleChange:D,values:l}),b&&e(Z,{outputValues:b,targetColumns:d})]})},q={mb:"1rem",mt:"1rem",ml:"5px",border:"1px solid rgba(0, 0, 0, 0.12)",padding:"1rem",borderRadius:"4px"},_=({deployment:a,publicDeployment:r=!1})=>c(g,{children:[e(T,{title:"README",children:e(i,{sx:q,children:e(F,{source:a.readme,warpperElement:{"data-color-mode":"light"}})})}),e(U,{deployment:a,publicDeployment:r}),(a==null?void 0:a.datasetSummary)&&e(T,{title:"Training Data",children:e(L,{columnsData:a.datasetSummary})})]}),V=({deployment:a})=>c(g,{children:[c(i,{sx:{display:"flex",alignItems:"center"},children:[e(A,{children:e(W,{mr:"auto",children:a.name})}),c(i,{sx:{display:"flex",flexDirection:"column",gap:"0.5rem",alignItems:"flex-end"},children:[e(i,{sx:{display:"flex",alignItems:"center",justifyContent:"center"},children:e(k,{sx:{textTransform:"uppercase",fontWeight:"bold",padding:"0.5rem"},status:a.status})}),e(i,{sx:{mb:"1rem",display:"flex",flexDirection:"row",alignItems:"center",gap:"0.5rem"},children:c(N,{children:["Rate Limit:"," ",e("b",{children:`${a.predictionRateLimitValue}/${a.predictionRateLimitUnit}`})]})})]})]}),e($,{sx:{mb:"1rem"}})]});export{V as D,_ as a};