import React from "react";
import { useLocation } from "react-router-dom";

const Results = () => {
  const location = useLocation();
  const { threshold, oldFilesCount, newFilesCount } = location.state || {};

  return (
    <main className="container mx-auto px-4 py-20">
      <h1 className="text-3xl font-bold mb-6">Analysis Results</h1>

      <div className="bg-white shadow-md rounded-lg p-6">
  <h2 className="text-lg font-semibold mb-4">Analysis Results</h2>
  <p className="text-sm text-gray-500 mb-2">Sample line-by-line feature comparison</p>

  <ul className="space-y-2 text-sm">
    <li className="flex items-center">
      <span className="text-green-600 font-semibold">✓ PASSED</span>
      <span className="ml-2 text-gray-700">
        Navigation menu: All 5 items found in expected positions
      </span>
    </li>

    {/* <li className="flex items-center">
      <span className="text-red-600 font-semibold">✗ FAILED</span>
      <span className="ml-2 text-gray-700">
        Login button: Text changed from "Sign In" to "Login" (similarity: 65%)
      </span>
    </li> */}

    <li className="flex items-center">
      <span className="text-green-600 font-semibold">✓ PASSED</span>
      <span className="ml-2 text-gray-700">
        Footer links: All social media icons preserved
      </span>
    </li>
  </ul>

  <a href="#" className="mt-4 inline-block text-green-600 hover:underline text-sm">
    Open full report with visual diffs
  </a>
</div>

    </main>
  );
};

export default Results;
