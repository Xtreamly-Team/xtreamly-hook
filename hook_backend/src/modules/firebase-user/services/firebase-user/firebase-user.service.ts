import { Injectable, NotFoundException } from '@nestjs/common';
import * as admin from 'firebase-admin';

export class FirebaseUser {
  public: string;
  private: string;
}

@Injectable()
export class FirebaseUserService {
  constructor() {
    const firebaseCreds = JSON.parse(process.env.FIREBASE_CREDENTIALS as string);
    admin.initializeApp({
      credential: admin.credential.cert(firebaseCreds),
    });

  }

  async findByWalletAddress(walletAddress: string): Promise<FirebaseUser> {
    const db = admin.firestore();
    const walletDoc = await db.collection("users")
        .doc(walletAddress)
        .collection("wallet")
        .doc("wallet")
        .get();

    if (!walletDoc.exists) {
      throw new NotFoundException(`User with wallet address ${walletAddress} not found.`);
    }

    const walletData = walletDoc.data();

    if (!walletData || !walletData.private) {
      throw new NotFoundException(`User with wallet address ${walletAddress} not found.`);
    }

    return {
      public: walletData.public,
      private: walletData.private,
    };
  }
}