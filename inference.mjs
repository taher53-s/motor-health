// Exported model values are immutable. This module performs no online training.
export function extractFeatures(samples){
 if(!Array.isArray(samples)||samples.length!==2048)throw new Error('Exactly 2048 acceleration samples are required.');
 if(samples.some(x=>typeof x!=='number'||!Number.isFinite(x)||Math.abs(x)>100))throw new Error('Samples must be finite numbers within ±100 source acceleration units.');
 const n=samples.length;let sum=0,square=0,peak=0;for(const x of samples){sum+=x;square+=x*x;peak=Math.max(peak,Math.abs(x))}const mean=sum/n,rms=Math.sqrt(square/n);let m2=0,m4=0;for(const x of samples){const d=x-mean;m2+=d*d;m4+=d**4}m2/=n;m4/=n;if(rms<1e-12||m2<1e-16)throw new Error('Flat or zero-energy signals cannot be scored.');return [rms,m4/(m2*m2),peak/rms];
}
export function scoreFeatures(f,model,name=model.selected){
 if(name==='RMS threshold')return f[0];
 const z=f.map((v,i)=>(v-model.mean[i])/model.scale[i]);
 if(name==='Autoencoder'){let a=z;for(let k=0;k<model.autoencoder.weights.length;k++){const w=model.autoencoder.weights[k],b=model.autoencoder.biases[k];a=b.map((v,j)=>{let sum=v;for(let i=0;i<a.length;i++)sum+=a[i]*w[i][j];return k===model.autoencoder.weights.length-1?sum:Math.tanh(sum)})}return a.reduce((sum,v,i)=>sum+(v-z[i])**2,0)/z.length}
 if(name==='Isolation Forest'){const Z=z.map(Math.fround);let total=0;for(const tree of model.isolation_forest.trees){let node=0,depth=0;while(tree.left[node]!==-1){node=Z[tree.feature[node]]<=tree.threshold[node]?tree.left[node]:tree.right[node];depth++}total+=depth+tree.leaf_c[node]}return 2**(-total/(model.isolation_forest.trees.length*model.isolation_forest.normalizer))}
 throw new Error('Unknown model');
}
export function infer(samples,model,name=model.selected){const f=extractFeatures(samples);const score=scoreFeatures(f,model,name);const threshold=model.thresholds[name];const ratio=score/threshold;const health=Math.max(0,Math.min(1,1-0.25*ratio**3));const band=health>=0.9?'healthy':health>=0.75?'watch':'inspect';return {health,band,health_scale:'Display index: clamp(1 - 0.25 × (raw score / threshold)^3, 0, 1); not a failure probability',model:name,model_version:model.version,features:Object.fromEntries(model.features.map((k,i)=>[k,f[i]])),score,threshold,ratio:score/threshold,prediction:score>=threshold?'anomaly':'normal',action:score>=threshold?'Review this window and arrange an inspection if operating context supports it.':'Continue monitoring; this score does not guarantee a healthy bearing.',scope:'CWRU benchmark; not a failure probability or live diagnosis'};}
