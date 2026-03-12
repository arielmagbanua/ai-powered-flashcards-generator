/**
 * Import function triggers from their respective submodules:
 *
 * import {onCall} from "firebase-functions/v2/https";
 * import {onDocumentWritten} from "firebase-functions/v2/firestore";
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

import { setGlobalOptions } from "firebase-functions";
import { onRequest } from "firebase-functions/https";
import * as logger from "firebase-functions/logger";
import Busboy from "busboy";

/**
 * AI Genkit packages
 */
import { googleAI } from "@genkit-ai/google-genai";
import { genkit, z } from "genkit";
import { PDFParse } from "pdf-parse";

// Start writing functions
// https://firebase.google.com/docs/functions/typescript

// For cost control, you can set the maximum number of containers that can be
// running at the same time. This helps mitigate the impact of unexpected
// traffic spikes by instead downgrading performance. This limit is a
// per-function limit. You can override the limit for each function using the
// `maxInstances` option in the function's options, e.g.
// `onRequest({ maxInstances: 5 }, (req, res) => { ... })`.
// NOTE: setGlobalOptions does not apply to functions using the v1 API. V1
// functions should each use functions.runWith({ maxInstances: 10 }) instead.
// In the v1 API, each function can only serve one request per container, so
// this will be the maximum concurrent request count.
setGlobalOptions({ maxInstances: 10 });

const generateFlashCards = async (content: string) => {
  // model schemas can be used to validate the response from the model and ensure that it is in the expected format.
  const inputSchema = z.object({
    content: z.string(),
  });
  const outputSchema = z.array(
    z.object({
      question: z.string(),
      answer: z.string(),
    }),
  );

  // create a genkit instance and specify the plugins to use. In this case, we are using the googleAI plugin to access Google's language models.
  const ai = genkit({
    plugins: [googleAI()],
  });

  // create the prompt flow and specify the input and output schemas for validation.
  const flashcardFlow = ai.defineFlow(
    {
      name: "Flashcard Flow",
      inputSchema: inputSchema,
      outputSchema: outputSchema,
    },
    async ({ content }) => {
      const { output } = await ai.generate({
        model: googleAI.model("gemini-3-flash-preview"),
        prompt: `
      You are a software engineer and a professor teaching about REST APIs.
      Generate a flashcards that will help students study and prepare for a quiz or exam.
      The flashcards format should be in json with the properties question, and answer.

      context: ${content}
      `,
      });

      return output;
    },
  );

  return await flashcardFlow({
    content: content,
  });
};

export const createFlashCards = onRequest(
  { secrets: ["GOOGLE_GENAI_API_KEY"] },
  async (request, response) => {
    // Ensure we're handling POST requests with multipart/form-data
    if (request.method !== "POST") {
      response.status(405).send("Method Not Allowed");
      return;
    }

    const busboy = Busboy({ headers: request.headers });
    let pdfBuffer: Buffer | null = null;

    busboy.on("file", (fieldname, file) => {
      if (fieldname === "pdfFile") {
        const chunks: Buffer[] = [];
        file.on("data", (data) => {
          chunks.push(data);
        });
        file.on("end", () => {
          pdfBuffer = Buffer.concat(chunks);
        });
      } else {
        file.resume();
      }
    });

    busboy.on("finish", async () => {
      if (!pdfBuffer) {
        response.status(400).send("Missing pdfFile in form submission");
        return;
      }

      try {
        // Initialize pdf-parse with the PDF buffer
        const parser = new PDFParse({ data: pdfBuffer });

        // Extract text content from the PDF
        const textResult = await parser.getText();
        const textContent = textResult.text;

        const result = await generateFlashCards(textContent);

        // Return the content of PDF immediately
        response.status(200).send(result);

        // genkit execution
      } catch (error) {
        logger.error("Error parsing PDF", error);
        response.status(500).send("Error processing PDF");
      }
    });

    // Handle errors in busboy stream
    busboy.on("error", (err) => {
      logger.error("Busboy error", err);
      response.status(500).send("Upload processing error");
    });

    // Node.js writeable stream needs the raw body
    // In Firebase v2, request.rawBody is available
    if ((request as any).rawBody) {
      busboy.end((request as any).rawBody);
    } else {
      request.pipe(busboy);
    }
  },
);
