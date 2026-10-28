import {useState} from "react";
const API=import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
type Msg={role:"user"|"assistant";content:string;sources?:{title:string;url:string}[]};
const starters=["What information is available about ITC Immunity Therapy Center?","Are treatment prices listed?","What does a treatment package include?"];
export default function App(){
 const [msgs,setMsgs]=useState<Msg[]>([]);
 const [question,setQuestion]=useState(""); const [busy,setBusy]=useState(false); const [error,setError]=useState("");
 async function send(text=question){
  const q=text.trim(); if(!q||busy)return;
  const previous=msgs; setMsgs(m=>[...m,{role:"user",content:q}]);setQuestion("");setBusy(true);setError("");
  try{
   const res=await fetch(`${API}/api/chat`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q,history:previous.slice(-8).map(m=>({role:m.role,content:m.content}))})});
   if(!res.ok)throw new Error("Chat service is unavailable. Please try again.");
   const data=await res.json();setMsgs(m=>[...m,{role:"assistant",content:data.answer,sources:data.sources||[]}]);
  }catch(e){setError(e instanceof Error?e.message:"Something went wrong.");}
  finally{setBusy(false);}
 }
 function reset(){setMsgs([]);setError("");setQuestion("");}
 return <main className="shell"><header><div className="mark">P</div><div><strong>PlacidWay</strong><small>Website Knowledge Assistant</small></div><button className="reset" onClick={reset}>New chat ↻</button></header>
 <section className="intro"><span className="eyebrow">PATIENT INFORMATION</span><h1>How can we help<br/>you explore your options?</h1><p>Ask about clinic information, treatment packages, or published prices. Answers use the approved website pages.</p></section>
 <section className="conversation" aria-live="polite">
 {!msgs.length&&<div className="suggestions">{starters.map(s=><button key={s} onClick={()=>send(s)}>{s}<span>↗</span></button>)}</div>}
 {msgs.map((m,i)=><article className={`message ${m.role}`} key={i}><b>{m.role==="user"?"You":"PlacidWay Assistant"}</b><p>{m.content}</p>{m.sources?.map((s,j)=><a key={j} href={s.url} target="_blank" rel="noreferrer">Source {j+1}: {s.title} ↗</a>)}</article>)}
 {busy&&<div className="typing">Reviewing approved sources…</div>}{error&&<p className="error">{error}</p>}
 </section>
 <form className="composer" onSubmit={e=>{e.preventDefault();void send();}}><textarea value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask a question about treatments, packages or prices…" maxLength={2000} rows={2}/><button disabled={busy||!question.trim()} aria-label="Send question">↑</button></form>
 <footer>Information assistant · Not a substitute for professional medical advice.</footer></main>;
}
