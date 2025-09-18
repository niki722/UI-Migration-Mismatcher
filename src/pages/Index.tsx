import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { FileUpload } from "@/components/ui/file-upload";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";


import { 
  Upload, 
  CheckCircle, 
  XCircle, 
  Zap, 
  Shield, 
  Database, 
  Eye, 
  BarChart3, 
  Download, 
  FileText, 
  GitCompare,
  Play,
  AlertCircle,
  Settings,
  Loader2
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [oldFiles, setOldFiles] = useState<File[]>([]);
  const [newFiles, setNewFiles] = useState<File[]>([]);
  const [similarityThreshold, setSimilarityThreshold] = useState([0.7]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const { toast } = useToast();
  const [analysisResults, setAnalysisResults] = useState<any[]>([]);
  const [showResults, setShowResults] = useState(false);



  const handleAnalyze = async () => {
  if (oldFiles.length === 0 || newFiles.length === 0) {
    toast({
      title: "Missing Files",
      description: "Please upload both old and new screenshots to begin analysis.",
      variant: "destructive",
    });
    return;
  }

  setIsAnalyzing(true);
  setAnalysisProgress(0);
  setShowResults(false);

  try {
    // Send files to backend /analyze
    const formData = new FormData();
    oldFiles.forEach(file => formData.append('old_files', file));
    newFiles.forEach(file => formData.append('new_files', file));

    const analyzeResponse = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      body: formData,
    });

    if (!analyzeResponse.ok) throw new Error("Analysis failed");

    setAnalysisProgress(50); // after sending files

    // Fetch summary as bullet points
    const summaryResponse = await fetch("http://127.0.0.1:5000/generate_summary");
    if (!summaryResponse.ok) throw new Error("Summary generation failed");

    const summaryData = await summaryResponse.json();
      // summaryData.summary is a string, split into lines
      const summaryLines = summaryData.summary.split('\n').filter(line => line.trim() !== "");
      setAnalysisResults(summaryLines.map(item => ({
        status: item.startsWith("PASSED") ? "PASSED" : item.startsWith("FAILED") ? "FAILED" : "INFO",
        message: item
      })));

    setAnalysisProgress(100);
    setShowResults(true);

    toast({
      title: "Analysis Complete!",
      description: `Successfully analyzed ${oldFiles.length + newFiles.length} screenshots.`,
    });

  } catch (error: any) {
    console.error(error);
    toast({
      title: "Analysis Failed",
      description: error.message || "Something went wrong.",
      variant: "destructive",
    });
  } finally {
    setIsAnalyzing(false);
    setTimeout(() => setAnalysisProgress(0), 2000);
  }
};


  const canAnalyze = oldFiles.length > 0 && newFiles.length > 0 && !isAnalyzing;

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-sm border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitCompare className="w-8 h-8 text-primary" />
            <span className="text-2xl font-bold">FeatureMirror</span>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <a href="#upload" className="text-muted-foreground hover:text-foreground transition-smooth">Upload</a>
            <a href="#features" className="text-muted-foreground hover:text-foreground transition-smooth">Features</a>
            <a href="#demo" className="text-muted-foreground hover:text-foreground transition-smooth">Demo</a>
            {/* <a href="#docs" className="text-muted-foreground hover:text-foreground transition-smooth">Docs</a> */}
            {/* <Button variant="outline" size="sm">Login</Button> */}
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 text-center">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="space-y-4">
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-hero bg-clip-text text-transparent animate-slide-up">
              Catch UI Migration Mismatches Instantly
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Upload before/after screenshots and get detailed feature-by-feature analysis. 
              No more manual QA hunting for broken elements after code migrations.
            </p>
          </div>
          
          {/* <div className="text-lg font-medium text-accent max-w-xl mx-auto border border-accent/20 rounded-lg p-4 bg-accent/5">
            **30-second elevator pitch:** Automated UI diff analysis that maps old→new screenshots, 
            extracts features with OCR, and gives you line-by-line Pass/Fail reports.
          </div> */}

          <div className="grid md:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <div className="flex items-center gap-3 text-left">
              <Zap className="w-6 h-6 text-accent shrink-0" />
              <span>Batch upload screenshots in seconds</span>
            </div>
            <div className="flex items-center gap-3 text-left">
              <Eye className="w-6 h-6 text-accent shrink-0" />
              <span>Auto-mapping with visual + text analysis</span>
            </div>
            <div className="flex items-center gap-3 text-left">
              <BarChart3 className="w-6 h-6 text-accent shrink-0" />
              <span>Human-readable Pass/Fail reports</span>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section id="upload" className="bg-muted/30 py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto">
            <div className="text-center space-y-4 mb-12">
              <h2 className="text-3xl md:text-4xl font-bold">Ready to Try FeatureMirror?</h2>
              <p className="text-lg text-muted-foreground">
                Upload your screenshots and let our AI analyze the differences
              </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-8 mb-8">
              <FileUpload
                title="Old Screenshots"
                description="Upload screenshots from your original UI version"
                files={oldFiles}
                onFilesChange={setOldFiles}
                maxFiles={10}
              />
              
              <FileUpload
                title="New Screenshots"
                description="Upload screenshots from your migrated UI version"
                files={newFiles}
                onFilesChange={setNewFiles}
                maxFiles={10}
              />
            </div>

            {/* <Card className="mb-8">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  Analysis Settings
                </CardTitle>
                <CardDescription>
                  Adjust the similarity threshold for feature matching
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="threshold">
                    Similarity Threshold: {(similarityThreshold[0] * 100).toFixed(0)}%
                  </Label>
                  <Slider
                    id="threshold"
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    value={similarityThreshold}
                    onValueChange={setSimilarityThreshold}
                    className="w-full"
                  />
                  <div className="text-xs text-muted-foreground grid grid-cols-3 gap-2">
                    <span>Strict (90%+): Catches minor changes</span>
                    <span>Balanced (70%): Recommended default</span>
                    <span>Loose (50%): Major changes only</span>
                  </div>
                </div>
              </CardContent>
            </Card> */}

            {isAnalyzing && (
              <Card className="mb-8">
                <CardContent className="p-6">
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="font-medium">Analyzing Screenshots...</span>
                    </div>
                    <Progress value={analysisProgress} className="w-full" />
                    <p className="text-sm text-muted-foreground">
                      Processing {oldFiles.length + newFiles.length} images with OCR and visual analysis
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            <div className="flex justify-center items-center mt-8">
            <Button 
              variant="hero" 
              size="lg" 
              onClick={handleAnalyze}
              disabled={!canAnalyze}
              className="min-w-[200px]"
            >
              {isAnalyzing ? (
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              ) : (
                <Play className="w-5 h-5 mr-2" />
              )}
              {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
            </Button>
          </div>
          <div className="flex justify-center items-center mt-8">

              {showResults && (
                <section className="py-20">
                  <div className="container mx-auto px-4">
                    <div className="max-w-4xl mx-auto">
                      {/* Header */}
                      <div className="text-center space-y-4 mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold">Analysis Results</h2>
                        <p className="text-lg text-muted-foreground">
                          Feature-by-feature comparison between old and new UI
                        </p>
                      </div>

                      {/* Summary */}
                      <div className="flex justify-center gap-6 mb-8">
                        <Badge variant="secondary" className="px-4 py-2 text-success font-medium">
                          ✅ {analysisResults.filter(r => r.status === "PASSED").length} Passed
                        </Badge>
                        <Badge variant="secondary" className="px-4 py-2 text-destructive font-medium">
                          ❌ {analysisResults.filter(r => r.status === "FAILED").length} Failed
                        </Badge>
                      </div>

                      {/* Results Card */}
                      <Card className="bg-card/50 backdrop-blur-sm">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <FileText className="w-5 h-5" />
                            Detailed Report
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-6">
                          {analysisResults.map((r, i) => (
                            <div
                              key={i}
                              className={`rounded-lg border p-4 space-y-2 ${
                                r.status === "PASSED"
                                  ? "border-success/30 bg-success/5"
                                  : "border-destructive/30 bg-destructive/5"
                              }`}
                            >
                              {/* Main Row */}
                              <div className="flex items-center gap-3">
                                {r.status === "PASSED" ? (
                                  <CheckCircle className="w-5 h-5 text-success" />
                                ) : (
                                  <XCircle className="w-5 h-5 text-destructive" />
                                )}
                                <span
                                  className={`font-semibold ${
                                    r.status === "PASSED" ? "text-success" : "text-destructive"
                                  }`}
                                >
                                  {r.status}
                                </span>
                                <span className="text-muted-foreground">→</span>
                                <span>{r.message}</span>
                              </div>

                              {/* Extra details for FAILED */}
                              {r.status === "FAILED" && r.details && (
                                <div className="ml-8 text-sm space-y-1 text-muted-foreground">
                                  <p><strong>Expected:</strong> {r.details.expected}</p>
                                  <p><strong>Found:</strong> {r.details.found}</p>
                                  {r.details.similarity && (
                                    <p><strong>Similarity:</strong> {r.details.similarity}%</p>
                                  )}
                                  {r.details.recommendation && (
                                    <p className="text-foreground">
                                      💡 <strong>Recommendation:</strong> {r.details.recommendation}
                                    </p>
                                  )}
                                </div>
                              )}
                            </div>
                          ))}

                  
                          {/* Download PDF link */}
                          <a
                            href="http://127.0.0.1:5000/download_report_pdf"
                            className="flex items-center gap-2 text-sm text-accent hover:text-accent-glow cursor-pointer transition-smooth"
                            download
                          >
                            <Download className="w-4 h-4" />
                            <span className="underline">Download PDF summary</span>
                          </a>


                        </CardContent>
                      </Card>
                    </div>
                  </div>
                </section>
              )}


              
              {!canAnalyze && !isAnalyzing && (
                <Alert className="max-w-md">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Upload both old and new screenshots to begin analysis
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center space-y-12">
            <h2 className="text-3xl md:text-4xl font-bold">How It Works</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Upload your before/after screenshots and let our **CPU-friendly open-source LLM** do the heavy lifting. 
              No GPU required, no data leaves your environment.
            </p>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                {
                  step: "1",
                  title: "Upload Screenshots",
                  description: "Batch upload multiple old and new screenshots. Supports common formats.",
                  icon: Upload
                },
                {
                  step: "2", 
                  title: "Auto-Map Features",
                  description: "Backend iterates through frames, extracts text & visual features, auto-maps old→new.",
                  icon: GitCompare
                },
                {
                  step: "3",
                  title: "LLM Analysis", 
                  description: "CPU-only open-source LLM (flan-t5-small) generates human-readable summaries.",
                  icon: BarChart3
                },
                {
                  step: "4",
                  title: "Review Results",
                  description: "Click any failed feature for full report with OCR text, visual diffs, remediation notes.",
                  icon: FileText
                }
              ].map((item, index) => (
                <Card key={index} className="text-left hover:shadow-feature transition-smooth">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                        {item.step}
                      </div>
                      <item.icon className="w-6 h-6 text-accent" />
                    </div>
                    <CardTitle className="text-lg">{item.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{item.description}</CardDescription>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features & Capabilities */}
      <section id="features" className="bg-muted/30 py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center space-y-4 mb-16">
              <h2 className="text-3xl md:text-4xl font-bold">Features & Capabilities</h2>
              <p className="text-lg text-muted-foreground">Built for developers, by developers</p>
            </div>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                {
                  title: "Batch Upload Support",
                  description: "Upload multiple old and new screenshots simultaneously",
                  icon: Upload
                },
                {
                  title: "Intelligent Auto-Mapping",
                  description: "Advanced algorithms match UI elements across versions",
                  icon: GitCompare
                },
                {
                  title: "OCR & Visual Diff",
                  description: "Extract text and compare visual elements pixel by pixel",
                  icon: Eye
                },
                {
                  title: "CPU-Only LLM",
                  description: "Fast summarization without expensive GPU requirements",
                  icon: Zap
                },
                {
                  title: "Downloadable Reports",
                  description: "Comprehensive PDF reports with detailed analysis results",
                  icon: Download
                },
                {
                  title: "Audit Logs",
                  description: "Complete traceability for compliance and debugging",
                  icon: Shield
                }
              ].map((feature, index) => (
                <Card key={index} className="hover:shadow-feature transition-smooth">
                  <CardHeader>
                    <feature.icon className="w-8 h-8 text-primary mb-2" />
                    <CardTitle>{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{feature.description}</CardDescription>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center space-y-4 mb-16">
              <h2 className="text-3xl md:text-4xl font-bold">Example Output</h2>
              <p className="text-lg text-muted-foreground">See what FeatureMirror reports look like</p>
            </div>
            
            <Card className="bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  Analysis Results
                </CardTitle>
                <CardDescription>Sample line-by-line feature comparison</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-background rounded-lg p-4 font-mono text-sm space-y-2">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-success" />
                    <span className="text-success">PASSED</span>
                    <span className="text-muted-foreground">→</span>
                    <span>Navigation menu: All 5 items found in expected positions</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <XCircle className="w-4 h-4 text-destructive" />
                    <span className="text-destructive">FAILED</span>
                    <span className="text-muted-foreground">→</span>
                    {/* <span>Login button: Text changed from "Sign In" to "Login" (similarity: 65%)</span> */}
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-success" />
                    <span className="text-success">PASSED</span>
                    <span className="text-muted-foreground">→</span>
                    <span>Footer links: All social media icons preserved</span>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 text-sm text-accent hover:text-accent-glow cursor-pointer transition-smooth">
                  <Download className="w-4 h-4" />
                  <span className="underline">Open full report with visual diffs</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Technical Stack */}
      <section className="bg-muted/30 py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center space-y-4 mb-16">
              <h2 className="text-3xl md:text-4xl font-bold">Technical Stack</h2>
              <p className="text-lg text-muted-foreground">Modern, reliable, developer-friendly</p>
            </div>
            
            <Card className="bg-gradient-card">
              <CardContent className="p-8">
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <h3 className="text-xl font-semibold">Backend</h3>
                    <div className="space-y-2 text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Database className="w-4 h-4" />
                        <span>FastAPI for REST endpoints</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Eye className="w-4 h-4" />
                        <span>Easyocr OCR for text extraction</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4" />
                        <span>Image hashing/SSIM for visual comparison</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h3 className="text-xl font-semibold">AI & Storage</h3>
                    <div className="space-y-2 text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Zap className="w-4 h-4" />
                        <span>flan-t5-small LLM (CPU-only)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Database className="w-4 h-4" />
                        <span>Optional database for persistence</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Shield className="w-4 h-4" />
                        <span>Local processing - images stay private</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="bg-muted/30 py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center space-y-4 mb-16">
              <h2 className="text-3xl md:text-4xl font-bold">Frequently Asked Questions</h2>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              {[
                {
                  question: "How accurate is the feature matching?",
                  answer: "Our visual + OCR matching typically achieves 85-95% accuracy."
                },
                {
                  question: "Do I need GPU for the LLM?",
                  answer: "No! We use CPU-only open-source models like flan-t5-small. Fast analysis without expensive GPU infrastructure."
                },
                {
                  question: "Are my images kept private?",
                  answer: "Yes. All processing happens locally on your infrastructure. Images are never sent to external services."
                },
                {
                  question: "Can I integrate with CI/CD?",
                  answer: "Absolutely. Full REST API with JSON responses. Perfect for automated testing pipelines and build processes."
                }
              ].map((faq, index) => (
                <Card key={index}>
                  <CardHeader>
                    <CardTitle className="text-lg">{faq.question}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-foreground">{faq.answer}</CardDescription>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </section>

      
    </main>
  );
};

export default Index;