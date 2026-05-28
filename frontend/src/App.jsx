import { useState, useEffect } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { TrendingUp, TrendingDown, Activity, AlertCircle } from 'lucide-react'
import axios from 'axios'

const API_BASE = "http://localhost:8000"

function App() {
  const [ticker, setTicker] = useState("AAPL")
  const [data, setData] = useState([])
  const [latestPrediction, setLatestPrediction] = useState(null)
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    let ignore = false;
    
    // Instantly clear the old chart when switching tickers for immediate UI feedback
    setData([])
    setLatestPrediction(null)
    setMetrics(null)

    // Poll data from FastAPI
    const fetchData = () => {
      // Always fetch historical to show the chart, our 'mode' toggle was just a label
      axios.get(`${API_BASE}/stream-data?ticker=${ticker}`).then(res => {
        if(!ignore && res.data && res.data.length > 0) {
          setData(res.data)
          
          const latestRecord = res.data[res.data.length - 1]
          if(latestRecord.price) {
            axios.post(`${API_BASE}/predict`, {
              ticker: ticker,
              model_name: "Stacking Ensemble",
              features: {
                price: latestRecord.price,
                volume: latestRecord.volume || 0,
                lag1: latestRecord.lag1 || 0,
                lag2: latestRecord.lag2 || 0,
                lag3: latestRecord.lag3 || 0,
                MA5: latestRecord.MA5 || latestRecord.price,
                MA10: latestRecord.MA10 || latestRecord.price,
                MA20: latestRecord.MA20 || latestRecord.price,
                rolling_mean: latestRecord.rolling_mean || latestRecord.price,
                rolling_std: latestRecord.rolling_std || 0,
                price_change: latestRecord.price_change || 0,
                pct_change: latestRecord.pct_change || 0
              }
            }).then(predRes => {
              if (!ignore) {
                setLatestPrediction(predRes.data)
              }
            }).catch(console.error)
          }
        }
      }).catch(console.error)
      
      axios.get(`${API_BASE}/model-metrics?ticker=${ticker}`).then(res => {
        if (!ignore && res.data && res.data.metrics) {
          setMetrics(res.data.metrics)
        }
      }).catch(console.error)
    }
    
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => {
      ignore = true;
      clearInterval(interval);
    }
  }, [ticker])

  const formatTime = (ts) => {
    if(!ts) return ""
    const d = new Date(ts)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const formatTooltipDate = (ts) => {
    if(!ts) return ""
    const d = new Date(ts)
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ', ' + 
           d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  // Calculate if the prediction is higher than current price
  const currentPrice = data.length > 0 ? data[data.length-1].price : 0;
  const isUpward = latestPrediction && latestPrediction.predicted_price > currentPrice;
  
  // If the last data point is older than 15 minutes, assume Market Closed
  const lastTimestamp = data.length > 0 ? new Date(data[data.length-1].timestamp).getTime() : Date.now();
  const isMarketClosed = (Date.now() - lastTimestamp) > (15 * 60 * 1000);

  return (
    <div className="min-h-screen bg-slate-50 p-6 md:p-10 font-sans text-gray-800">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row gap-8">
        
        {/* Main Content (Left/Center) */}
        <div className="flex-1 space-y-8">
        
        {/* Simple Header */}
        <header className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold tracking-tight text-gray-900">Stock ML Predictor</h1>
              <span className={`px-2.5 py-1 text-xs font-bold rounded-full ${isMarketClosed ? 'bg-gray-200 text-gray-600' : 'bg-green-100 text-green-700 animate-pulse'}`}>
                {isMarketClosed ? 'Market Closed' : 'Market Open (Live)'}
              </span>
            </div>
            <p className="text-gray-500 mt-1">Simple, clean market forecasts.</p>
          </div>
          
          <select 
            value={ticker} onChange={e => setTicker(e.target.value)}
            className="bg-white border hover:border-gray-300 text-gray-700 text-lg rounded-xl shadow-sm focus:ring-blue-500 block px-4 py-2 outline-none transition-all cursor-pointer"
          >
            <option value="AAPL">Tech: Apple (AAPL)</option>
            <option value="TSLA">Auto: Tesla (TSLA)</option>
            <option value="GOOG">Tech: Google (GOOG)</option>
            <option value="INFY">IT: Infosys (INFY)</option>
          </select>
        </header>

        {/* Big Minimal Metric */}
        <div className="bg-white rounded-3xl p-8 border border-gray-100 shadow-sm flex flex-col md:flex-row items-center justify-between gap-8">
          <div>
            <p className="text-sm font-medium text-gray-400 tracking-wide uppercase">Current Price</p>
            <h2 className="text-5xl font-bold text-gray-900 mt-1">
              ${currentPrice ? currentPrice.toFixed(2) : "0.00"}
            </h2>
          </div>

          <div className="h-16 w-px bg-gray-100 hidden md:block"></div>

          <div>
            <p className="text-sm font-medium text-gray-400 tracking-wide uppercase">ML Predicted Price</p>
            <div className="flex items-center gap-3 mt-1">
              <h2 className={`text-5xl font-bold ${isUpward ? 'text-green-600' : 'text-red-500'}`}>
                ${latestPrediction ? latestPrediction.predicted_price.toFixed(2) : "0.00"}
              </h2>
              {isUpward ? <TrendingUp className="w-8 h-8 text-green-500" /> : <TrendingDown className="w-8 h-8 text-red-500" />}
            </div>
          </div>
        </div>

        {/* Minimal Explanation */}
        <div className="bg-blue-50/50 rounded-2xl p-6 text-blue-800 flex gap-4 items-start">
          <AlertCircle className="w-6 h-6 flex-shrink-0 mt-0.5 text-blue-500" />
          <p className="text-lg/relaxed">
            Based on recent momentum, our ML model predicts the price will 
            <span className="font-bold"> {isUpward ? "go UP" : "go DOWN"} </span> 
            in the immediate future. The graph below displays the actual price history on the left, and visually extends to show where the ML model expects the price to be next.
          </p>
        </div>

        {/* Simplified Chart */}
        <div className="bg-white rounded-3xl shadow-sm border border-gray-100 p-6">
          <div className="h-[400px] w-full">
            {data.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                  <XAxis dataKey="timestamp" tickFormatter={formatTime} tick={{fill: '#9ca3af', fontSize: 13}} dy={10} axisLine={false} tickLine={false} />
                  <YAxis domain={['auto', 'auto']} tick={{fill: '#9ca3af', fontSize: 13}} dx={-10} tickFormatter={(val) => `$${val}`} axisLine={false} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '16px', border: '1px solid #f3f4f6', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                    labelFormatter={formatTooltipDate}
                  />
                  <Area type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorPrice)" name="Actual Price" />
                  
                  {/* Visual marker for current price */}
                  <ReferenceLine y={currentPrice} stroke="#9ca3af" strokeDasharray="3 3" />
                  
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-400">Loading live stock data...</div>
            )}
          </div>
        </div>
        </div>

        {/* Right Sidebar for Metrics */}
        <div className="w-full lg:w-80 flex-shrink-0 space-y-6">
          <div className="bg-white rounded-3xl p-6 border border-gray-100 shadow-sm">
            <div className="flex items-center gap-2 mb-6">
              <Activity className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-bold text-gray-900">Model Accuracy</h3>
            </div>
            
            {!metrics ? (
              <div className="text-gray-400 text-sm text-center py-4">Loading metrics...</div>
            ) : (
              <div className="space-y-4">
                {Object.entries(metrics).map(([modelName, accuracy]) => {
                  const isTop = accuracy === Math.max(...Object.values(metrics).map(a => parseFloat(a) || 0)) + "%";
                  return (
                    <div key={modelName} className="flex flex-col gap-1">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-600">{modelName}</span>
                        <span className={`text-sm font-bold ${parseFloat(accuracy) > 90 ? 'text-green-600' : 'text-gray-900'}`}>{accuracy}</span>
                      </div>
                      <div className="w-full bg-gray-100 rounded-full h-1.5">
                        <div 
                          className={`h-1.5 rounded-full ${parseFloat(accuracy) > 90 ? 'bg-green-500' : 'bg-blue-500'}`} 
                          style={{ width: accuracy === "N/A" ? "0%" : accuracy }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          
          <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-3xl p-6 text-white shadow-md">
            <h3 className="font-bold mb-2">Live Ensembling</h3>
            <p className="text-blue-100 text-sm leading-relaxed">
              Our system runs {metrics ? Object.keys(metrics).length - 2 : 4} base models simultaneously in the background without Apache Spark. The final prediction uses a Stacking Ensemble method, combining all insights for maximum accuracy.
            </p>
          </div>
        </div>

      </div>
    </div>
  )
}

export default App
